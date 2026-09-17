"""
ConceptAttention for ComfyUI.

Real implementation of "ConceptAttention: Diffusion Transformers Learn Highly
Interpretable Features" (Helbling et al., arXiv:2502.04320).

Concept tokens run through a parallel residual stream that reuses each model's
own text attention weights. The saliency map for a concept is the linear
projection between the concept attention output vectors and the image attention
output vectors in the attention output space, softmaxed across concepts.

Both entry points share one engine:

  * encode  - one forward pass over an existing image (`compute_concept_attention`)
  * generate - hooks installed on a patched MODEL, collected across every
    denoising step, then turned into heatmaps

Capture is observational everywhere, so it never changes what the model
produces and quantized (nvfp4 / int8-convrot) weights go through the normal
ComfyUI ops.

Supported backbones:
  * Flux.2 / Flux.2 Klein  -> comfy.ldm.flux.model.Flux with global_modulation
  * Krea 2                 -> comfy.ldm.krea2.model.SingleStreamDiT
"""

from dataclasses import dataclass, field

import torch

import comfy.model_base
import comfy.model_management
import comfy.patcher_extension as pe
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import attention as flux_attention
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


@dataclass
class ConceptMaps:
    """Concept saliency maps, normalized to [0, 1] at image resolution."""
    concepts: list = field(default_factory=list)
    maps: torch.Tensor = None          # (C, H, W) float32 cpu
    height: int = 0
    width: int = 0

    @property
    def num_concepts(self):
        return len(self.concepts)


class ConceptAttentionState:
    """Shared, mutable accumulator for a single concept attention run."""

    def __init__(self, concepts, concept_ctx, layer_indices, softmax=True, temperature=1.0):
        self.concepts = concepts
        self.concept_ctx = concept_ctx
        self.cache = set(layer_indices)
        self.softmax = softmax
        self.temperature = temperature

        self.dit = None
        self.grid = None
        self.residual = None
        self.concept_pe = None
        self.tvec = None
        self.call_concept = {}
        self.scores_sum = None
        self.count = 0

    @property
    def num_concepts(self):
        return len(self.concepts)

    # -- per-forward setup -------------------------------------------------

    def begin_call(self, dit, x, timesteps, is_krea2):
        self.dit = dit
        self.grid = _grid_shape(dit, x)
        self.residual = self._initial_residual(dit, x, is_krea2)
        axes = 3 if is_krea2 else len(dit.params.axes_dim)
        ids = torch.zeros(x.shape[0], self.num_concepts, axes, device=x.device, dtype=torch.float32)
        self.concept_pe = dit.pe_embedder(ids)
        if is_krea2:
            t = dit.tmlp(timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(x.dtype))
            self.tvec = dit.tproj(t)

    def end_call(self):
        self.tvec = None
        self.call_concept = {}

    def _initial_residual(self, dit, x, is_krea2):
        ctx = comfy.model_management.cast_to_device(self.concept_ctx, x.device, x.dtype)
        if is_krea2:
            packed = ctx.reshape(1, self.num_concepts, dit.txtlayers, dit.txtdim)
            residual = dit.txtmlp(dit.txtfusion(packed, mask=None, transformer_options={}))
        else:
            residual = dit.txt_in(ctx)
        return residual.expand(x.shape[0], -1, -1)

    def add_scores(self, dot, layer_index):
        if layer_index not in self.cache:
            return
        dot = dot.detach().float().cpu()
        self.scores_sum = dot if self.scores_sum is None else self.scores_sum + dot
        self.count += 1


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def _raw_token_ids(clip, text):
    tokens = clip.tokenize(text)
    key = next(iter(tokens))
    return [t[0] for t in tokens[key][0]], tokens


def _concept_span(token_ids):
    """Return the [start, end) span of the concept text inside the chat template."""
    user_turns = [i for i in range(len(token_ids) - 1) if token_ids[i] == 151644 and token_ids[i + 1] == 872]
    if not user_turns:
        return 0, len(token_ids)
    start = user_turns[0] + 3
    end = next((i for i in range(start, len(token_ids)) if token_ids[i] == 151645), len(token_ids))
    return start, end


def _encode_context(clip, text):
    token_ids, tokens = _raw_token_ids(clip, text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    return cond[0][0], token_ids


def encode_concept_context(clip, concepts, strip_prefix):
    """Encode each concept and average its token span into a single embedding row."""
    vectors = []
    for concept in concepts:
        context, token_ids = _encode_context(clip, concept)
        start, end = _concept_span(token_ids)
        lo, hi = (0, end - start) if strip_prefix else (start, end)
        if hi <= lo:
            lo, hi = 0, 1
        vectors.append(context[:, lo:hi].mean(dim=1).squeeze(0))
    return torch.stack(vectors, dim=0).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def depth_of(dit):
    return len(dit.double_blocks) if hasattr(dit, "double_blocks") else len(dit.blocks)


def _layer_indices_for(dit, layer_start, layer_end):
    depth = depth_of(dit)
    if layer_start is None or layer_start < 0:
        layer_start = max(0, depth - 4)
    if layer_end is None or layer_end <= 0:
        layer_end = depth
    layer_start = max(0, min(layer_start, depth - 1))
    layer_end = max(layer_start + 1, min(layer_end, depth))
    return list(range(layer_start, layer_end))


def _grid_shape(dit, latent):
    patch = getattr(dit, "patch", None) or getattr(dit, "patch_size", 1)
    steps_h = (latent.shape[-2] + (patch // 2)) // patch
    steps_w = (latent.shape[-1] + (patch // 2)) // patch
    return steps_h, steps_w


def _cond_index(transformer_options):
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if not cond_or_uncond:
        return 0
    try:
        return cond_or_uncond.index(0)
    except ValueError:
        return None


def _repeat_kv(t, heads):
    if t.shape[1] == heads:
        return t
    return t.repeat_interleave(heads // t.shape[1], dim=1)


def _is_supported(base):
    dit = base.diffusion_model
    if isinstance(base, comfy.model_base.Krea2):
        return dit, True
    if getattr(getattr(dit, "params", None), "global_modulation", False):
        return dit, False
    raise NotImplementedError("ConceptAttention supports Flux.2 / Flux.2 Klein and Krea 2 models")


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------

def install(state, dit, is_krea2, transformer_options):
    """Register the observational capture hooks on a transformer_options dict."""
    if is_krea2:
        patches = transformer_options.setdefault("patches", {})
        patches.setdefault("attn1_patch", []).append(_krea2_attn_patch)
        patches.setdefault("attn1_output_patch", []).append(_krea2_output_patch)
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "concept_attention", _krea2_wrapper, transformer_options)
    else:
        replace = transformer_options.setdefault("patches_replace", {}).setdefault("dit", {})
        for index in range(depth_of(dit)):
            replace[("double_block", index)] = _flux2_block_replacement(index)
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "concept_attention", _flux2_wrapper, transformer_options)


# ---------------------------------------------------------------------------
# Flux.2 / Flux.2 Klein
# ---------------------------------------------------------------------------

def _flux2_wrapper(executor, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs):
    state = transformer_options.get("concept_state")
    if state is not None:
        state.begin_call(executor.class_obj, x, timestep, is_krea2=False)
    try:
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)
    finally:
        if state is not None:
            state.end_call()


def _mod_pair(vec, block):
    if isinstance(vec, tuple):
        return vec[0], vec[1]
    return block.img_mod(vec), block.txt_mod(vec)


def _flux2_block_replacement(layer_index):
    """Double-block wrapper that also advances and scores the concept stream."""
    def replacement(args, extra):
        transformer_options = args.get("transformer_options", {})
        state = transformer_options.get("concept_state")
        if state is None or state.residual is None:
            return extra["original_block"](args)

        block = state.dit.double_blocks[layer_index]
        img, txt, vec, pe = args["img"], args["txt"], args["vec"], args["pe"]
        img_mods, txt_mods = _mod_pair(vec, block)
        (img_mod1, _), (txt_mod1, txt_mod2) = img_mods, txt_mods
        txt_len = txt.shape[1]

        img_modulated = block.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = block.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = block.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = block.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        img_attn = flux_attention(q, k, v, pe=pe)[:, txt_len:]

        concepts = state.residual
        concept_modulated = block.txt_norm1(concepts)
        concept_modulated = (1 + txt_mod1.scale) * concept_modulated + txt_mod1.shift
        concept_qkv = block.txt_attn.qkv(concept_modulated)
        concept_q, concept_k, concept_v = concept_qkv.view(concept_qkv.shape[0], concept_qkv.shape[1], 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
        concept_q, concept_k = block.txt_attn.norm(concept_q, concept_k, concept_v)

        cq = torch.cat((concept_q, img_q), dim=2)
        ck = torch.cat((concept_k, img_k), dim=2)
        cv = torch.cat((concept_v, img_v), dim=2)
        concept_pe = torch.cat((state.concept_pe, pe[:, :, txt_len:]), dim=2)
        concept_attn = flux_attention(cq, ck, cv, pe=concept_pe)[:, :state.num_concepts]

        index = _cond_index(transformer_options)
        if index is not None:
            state.add_scores((concept_attn @ img_attn.transpose(1, 2))[index], layer_index)

        concepts = concepts + txt_mod1.gate * block.txt_attn.proj(concept_attn)
        concept_res = block.txt_norm2(concepts)
        concept_res = (1 + txt_mod2.scale) * concept_res + txt_mod2.shift
        concepts = concepts + txt_mod2.gate * block.txt_mlp(concept_res)
        state.residual = concepts

        # the original block mutates img/txt in place, so it runs last
        return extra["original_block"](args)

    return replacement


# ---------------------------------------------------------------------------
# Krea 2
# ---------------------------------------------------------------------------

def _krea2_wrapper(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
    state = transformer_options.get("concept_state")
    if state is not None:
        state.begin_call(executor.class_obj, x, timesteps, is_krea2=True)
    try:
        return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
    finally:
        if state is not None:
            state.end_call()


def _krea2_attn_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
    extra_options = extra_options or {}
    state = extra_options.get("concept_state")
    if state is None or extra_options.get("block_type") != "single" or "img_slice" not in extra_options:
        return {}
    if state.residual is None or state.tvec is None:
        return {}

    layer_index = extra_options["block_index"]
    txt_len, total = extra_options["img_slice"]
    block = state.dit.blocks[layer_index]
    attn = block.attn
    heads = attn.heads

    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(state.tvec)
    cpre = (1 + prescale) * block.prenorm(state.residual) + preshift
    cq = attn.wq(cpre).view(cpre.shape[0], cpre.shape[1], heads, -1).transpose(1, 2)
    ck = attn.wk(cpre).view(cpre.shape[0], cpre.shape[1], attn.kvheads, -1).transpose(1, 2)
    cv = attn.wv(cpre).view(cpre.shape[0], cpre.shape[1], attn.kvheads, -1).transpose(1, 2)
    cgate = attn.gate(cpre)
    cq, ck = attn.qknorm(cq, ck)

    cq, ck = apply_rope(cq, ck, state.concept_pe)
    _, roped_k = apply_rope(q, k, pe)
    ck = torch.cat((ck, roped_k[:, :, txt_len:total]), dim=2)
    cv = torch.cat((cv, v[:, :, txt_len:total]), dim=2)
    concept_attn = optimized_attention_masked(cq, _repeat_kv(ck, heads), _repeat_kv(cv, heads), heads, skip_reshape=True)

    state.call_concept[layer_index] = concept_attn.detach()
    state.residual = state.residual + pregate * attn.wo(concept_attn * torch.sigmoid(cgate))
    state.residual = state.residual + postgate * block.mlp((1 + postscale) * block.postnorm(state.residual) + postshift)
    return {}


def _krea2_output_patch(out, extra_options):
    state = extra_options.get("concept_state")
    if state is None or extra_options.get("block_type") != "single" or "img_slice" not in extra_options:
        return out
    layer_index = extra_options["block_index"]
    concept = state.call_concept.pop(layer_index, None)
    if concept is None:
        return out
    txt_len, total = extra_options["img_slice"]
    img_attn = out[:, txt_len:total]
    index = _cond_index(extra_options)
    if index is not None:
        state.add_scores((concept @ img_attn.transpose(1, 2))[index], layer_index)
    return out


# ---------------------------------------------------------------------------
# Encode entry point
# ---------------------------------------------------------------------------

def make_state(model, clip, concepts, layer_start, layer_end, softmax=True, temperature=1.0):
    """Build a ConceptAttentionState plus the resolved backbone for the nodes."""
    base = model.model
    dit, is_krea2 = _is_supported(base)
    concepts = [c.strip() for c in concepts if c and c.strip()]
    if not concepts:
        raise ValueError("concept_list must contain at least one concept")
    layer_indices = _layer_indices_for(dit, layer_start, layer_end)
    concept_ctx = encode_concept_context(clip, concepts, is_krea2)
    return ConceptAttentionState(concepts, concept_ctx, layer_indices, softmax, temperature), dit, is_krea2


def compute_concept_attention(model, vae, clip, image, prompt, concepts,
                              layer_start=None, layer_end=None, num_steps=4,
                              noise_timestep=2, seed=0, softmax=True,
                              temperature=1.0):
    base = model.model
    state, dit, is_krea2 = make_state(model, clip, concepts, layer_start, layer_end, softmax, temperature)

    latent = vae.encode(image[..., :3])
    latent = base.process_latent_in(latent)

    t_value = float(base.model_sampling.percent_to_sigma(min(max(noise_timestep / max(num_steps, 1), 0.0), 0.999)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(latent.device).to(latent.dtype)
    noisy = (1.0 - t_value) * latent + t_value * noise
    t = torch.full((latent.shape[0],), t_value, device=latent.device, dtype=latent.dtype)

    context, _ = _encode_context(clip, prompt)

    comfy.model_management.load_model_gpu(model)
    device = getattr(model, "load_device", None) or comfy.model_management.get_torch_device()
    noisy = noisy.to(device)

    transformer_options = {}
    if "transformer_options" in model.model_options:
        transformer_options = pe.merge_nested_dicts(transformer_options, model.model_options["transformer_options"], copy_dict1=False)
    transformer_options["concept_state"] = state
    install(state, dit, is_krea2, transformer_options)

    base.apply_model(noisy, t, c_crossattn=context, transformer_options=transformer_options)
    return build_maps(state, image)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def build_maps(state, image):
    if state.scores_sum is None:
        raise RuntimeError("ConceptAttention did not collect any attention; was the patched model used?")

    scores = state.scores_sum / max(state.count, 1)
    if state.softmax:
        scores = torch.softmax(scores / max(float(state.temperature), 1e-6), dim=0)

    grid_h, grid_w = state.grid
    if grid_h * grid_w != scores.shape[1]:
        raise RuntimeError(f"ConceptAttention image tokens ({scores.shape[1]}) do not match latent grid {grid_h}x{grid_w}")

    out_h, out_w = image.shape[1], image.shape[2]
    maps = scores.reshape(scores.shape[0], grid_h, grid_w).unsqueeze(1)
    maps = torch.nn.functional.interpolate(maps, size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(1)

    flat = maps.reshape(maps.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    maps = ((flat - lo) / (hi - lo + 1e-8)).reshape(maps.shape).float()

    return ConceptMaps(list(state.concepts), maps, out_h, out_w)
