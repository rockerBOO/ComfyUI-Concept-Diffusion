"""
ComfyUI nodes for ConceptAttention.

Give it an image, a Flux.2 (Klein/dev) or Krea 2 model, a VAE, a CLIP, a prompt
and a list of concepts. It returns a per-concept saliency grid and an overlay.
"""

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps

from .concept_attention import ConceptMaps, compute_concept_attention

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
    array = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return torch.from_numpy(array)[None, ...]


def _colormap(heatmap):
    rgba = colormaps["plasma"](np.asarray(heatmap, dtype=np.float32))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _label(image, text):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, image.height - 26, image.width, image.height), fill=(0, 0, 0))
    draw.text((6, image.height - 22), text, fill=(255, 255, 255))
    return np.asarray(image)


def _tile_heatmaps(maps, concepts):
    panels = [_label(_colormap(maps[i]), concepts[i]) for i in range(len(concepts))]
    return np.concatenate(panels, axis=1)


def _overlay_map(image, heatmap, alpha):
    colored = colormaps["plasma"](np.asarray(heatmap, dtype=np.float32))[:, :, :3]
    return np.clip(image * (1.0 - alpha) + colored * alpha, 0.0, 1.0)


def _overlay_concepts(image, maps, alpha):
    overlay = image.copy()
    colors = colormaps["tab10"](np.linspace(0, 1, max(len(maps), 1)))[:, :3]
    if len(maps) == 1:
        colors = np.array([[1.0, 0.2, 0.0]])
    for i in range(len(maps)):
        m = np.asarray(maps[i], dtype=np.float32)[:, :, None]
        overlay = overlay * (1.0 - alpha * m) + colors[i] * (alpha * m)
    return np.clip(overlay, 0.0, 1.0)


class ConceptAttentionNode:
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
                "layer_start": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "layer_end": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "softmax": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def run(self, model, vae, clip, image, prompt, concepts, noise_timestep, num_steps,
            layer_start, layer_end, softmax, temperature, alpha, seed):
        names = _parse_concepts(concepts)
        maps = compute_concept_attention(
            model, vae, clip, image, prompt, names,
            layer_start=layer_start, layer_end=layer_end, num_steps=num_steps,
            noise_timestep=noise_timestep, seed=seed, softmax=softmax,
            temperature=temperature,
        )
        logger.info("ConceptAttention: %d concepts over %dx%d", maps.num_concepts, maps.width, maps.height)

        base = _to_numpy(image)
        heatmaps = _tile_heatmaps(maps.maps.numpy(), maps.concepts)
        overlay = _overlay_concepts(base, maps.maps.numpy(), alpha)
        return maps, _to_comfy_image(heatmaps), _to_comfy_image(overlay)


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
            overlay = _overlay_map(base, concept_maps.maps[concept_maps.concepts.index(name)].numpy(), alpha)
        else:
            overlay = _overlay_concepts(base, concept_maps.maps.numpy(), alpha)
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
    "ConceptAttentionVisualizerNode": ConceptAttentionVisualizerNode,
    "ConceptSaliencyMapNode": ConceptSaliencyMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptAttentionNode": "Concept Attention",
    "ConceptAttentionVisualizerNode": "Concept Attention Visualizer",
    "ConceptSaliencyMapNode": "Concept Saliency Map",
}
