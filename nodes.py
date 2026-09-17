"""
ComfyUI nodes for ConceptAttention.

Two ways to use it:

  * Concept Attention (encode) - attribute an existing image in one forward pass.
  * Concept Attention Model (generate) - patch a MODEL, sample normally, then
    Concept Attention Maps turns the collected attention into heatmaps for the
    generated image.
"""

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps

from .concept_attention import (
    ConceptMaps,
    build_maps,
    compute_concept_attention,
    install,
    make_state,
    resolve_layers,
)

logger = logging.getLogger(__name__)


def _parse_concepts(text):
    return [c.strip() for c in text.replace("\n", ",").split(",") if c.strip()]


def _to_numpy(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]
    return np.clip(image, 0.0, 1.0)


def _to_comfy_image(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size and float(array.max()) > 1.5:
        array = array / 255.0
    return torch.from_numpy(np.clip(array, 0.0, 1.0))[None, ...]


def _colormap(heatmap):
    rgba = colormaps["plasma"](np.asarray(heatmap, dtype=np.float32))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _label(image, text):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, image.height - 26, image.width, image.height), fill=(0, 0, 0))
    draw.text((6, image.height - 22), text, fill=(255, 255, 255))
    return np.asarray(image)


def _visualize(maps, image, alpha):
    heatmaps = np.concatenate([_label(_colormap(maps.maps[i]), maps.concepts[i]) for i in range(maps.num_concepts)], axis=1)

    base = _to_numpy(image)
    overlay = base.copy()
    colors = colormaps["tab10"](np.linspace(0, 1, max(maps.num_concepts, 1)))[:, :3]
    if maps.num_concepts == 1:
        colors = np.array([[1.0, 0.2, 0.0]])
    for i in range(maps.num_concepts):
        m = maps.maps[i].numpy()[:, :, None]
        overlay = overlay * (1.0 - alpha * m) + colors[i] * (alpha * m)
    return _to_comfy_image(heatmaps), _to_comfy_image(np.clip(overlay, 0.0, 1.0))


_COMMON_CONCEPT_WIDGETS = {
    "layer_start": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
    "layer_end": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
    "softmax": ("BOOLEAN", {"default": True}),
    "temperature": ("FLOAT", {"default": 1000.0, "min": 0.01, "max": 100000.0, "step": 0.01}),
}


class ConceptAttentionNode:
    """One-shot concept attention over an existing image."""

    CATEGORY = "Concept Attention"
    RETURN_TYPES = ("CONCEPT_MAPS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("concept_maps", "heatmaps", "overlay")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A dragon on a hill."}),
                "concepts": ("STRING", {"multiline": True, "default": "dragon, rock, sky, clouds"}),
                "noise_timestep": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "num_steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                **_COMMON_CONCEPT_WIDGETS,
            }
        }

    def run(self, model, vae, clip, image, prompt, concepts, noise_timestep, num_steps,
            alpha, seed, layer_start, layer_end, softmax, temperature):
        maps = compute_concept_attention(
            model, vae, clip, image, prompt, _parse_concepts(concepts),
            layer_start=layer_start, layer_end=layer_end, num_steps=num_steps,
            noise_timestep=noise_timestep, seed=seed, softmax=softmax,
            temperature=temperature,
        )
        logger.info("ConceptAttention: %d concepts over %dx%d", maps.num_concepts, maps.width, maps.height)
        heatmaps, overlay = _visualize(maps, image, alpha)
        return maps, heatmaps, overlay


class ConceptAttentionModel:
    """Patch a MODEL so concept attention is collected during normal sampling."""

    CATEGORY = "Concept Attention"
    RETURN_TYPES = ("MODEL", "CONCEPT_ATTENTION")
    RETURN_NAMES = ("model", "concept_state")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "concepts": ("STRING", {"multiline": True, "default": "dragon, rock, sky, clouds"}),
            }
        }

    def apply(self, model, clip, concepts):
        state, dit, is_krea2 = make_state(model, clip, _parse_concepts(concepts))
        patcher = model.clone()
        transformer_options = patcher.model_options.setdefault("transformer_options", {})
        transformer_options["concept_state"] = state
        install(state, dit, is_krea2, transformer_options)
        return patcher, state


class ConceptAttentionMaps:
    """Turn the attention collected during sampling into heatmaps."""

    CATEGORY = "Concept Attention"
    RETURN_TYPES = ("CONCEPT_MAPS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("concept_maps", "heatmaps", "overlay")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_state": ("CONCEPT_ATTENTION",),
                "image": ("IMAGE",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                **_COMMON_CONCEPT_WIDGETS,
            }
        }

    def run(self, concept_state, image, alpha, layer_start, layer_end, softmax, temperature):
        maps = build_maps(concept_state, image, resolve_layers(concept_state, layer_start, layer_end), softmax, temperature)
        heatmaps, overlay = _visualize(maps, image, alpha)
        return maps, heatmaps, overlay


class ConceptAttentionVisualizerNode:
    CATEGORY = "Concept Attention"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("CONCEPT_MAPS",),
                "image": ("IMAGE",),
                "concept_name": ("STRING", {"default": ""}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def run(self, concept_maps, image, concept_name, alpha):
        base = _to_numpy(image)
        name = concept_name.strip()
        if name and name in concept_maps.concepts:
            index = concept_maps.concepts.index(name)
            colored = colormaps["plasma"](concept_maps.maps[index].numpy())[:, :, :3]
            overlay = np.clip(base * (1.0 - alpha) + colored * alpha, 0.0, 1.0)
        else:
            _, overlay = _visualize(concept_maps, image, alpha)
            return (overlay,)
        return (_to_comfy_image(overlay),)


class ConceptSaliencyMapNode:
    CATEGORY = "Concept Attention"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "saliency")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "concept_maps": ("CONCEPT_MAPS",),
                "concept_name": ("STRING", {"default": ""}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def run(self, concept_maps, concept_name, threshold):
        if not concept_maps.concepts:
            raise ValueError("concept_maps is empty")
        name = concept_name.strip()
        index = concept_maps.concepts.index(name) if name in concept_maps.concepts else 0
        heatmap = concept_maps.maps[index].numpy()
        mask = (heatmap > threshold).astype(np.float32)
        colored = _colormap(heatmap)
        colored[mask < 0.5] = 0
        return torch.from_numpy(mask)[None, ...], _to_comfy_image(colored)


NODE_CLASS_MAPPINGS = {
    "ConceptAttentionNode": ConceptAttentionNode,
    "ConceptAttentionModel": ConceptAttentionModel,
    "ConceptAttentionMaps": ConceptAttentionMaps,
    "ConceptAttentionVisualizerNode": ConceptAttentionVisualizerNode,
    "ConceptSaliencyMapNode": ConceptSaliencyMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptAttentionNode": "Concept Attention (encode image)",
    "ConceptAttentionModel": "Concept Attention Model (generate)",
    "ConceptAttentionMaps": "Concept Attention Maps",
    "ConceptAttentionVisualizerNode": "Concept Attention Visualizer",
    "ConceptSaliencyMapNode": "Concept Saliency Map",
}
