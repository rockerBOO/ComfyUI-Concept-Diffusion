"""
ConceptAttention for ComfyUI.

Real implementation of "ConceptAttention: Diffusion Transformers Learn Highly
Interpretable Features" (Helbling et al., arXiv:2502.04320).

Concept tokens run through a parallel residual stream that reuses each model's
own text attention weights. The saliency map for a concept is the linear
projection between the concept attention output vectors and the image attention
output vectors in the attention output space, softmaxed across concepts.

Supported backbones (comfy):
  * Flux.2 / Flux.2 Klein  -> comfy.ldm.flux.model.Flux with global_modulation
  * Krea 2                 -> comfy.ldm.krea2.model.SingleStreamDiT

The model forward is re-run here with the model's own (possibly quantized)
modules, so nvfp4 / int8-convrot weights go through the normal ComfyUI ops.
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

    def as_dict(self):
        return {name: self.maps[i] for i, name in enumerate(self.concepts)}

    @property
    def num_concepts(self):
        return len(self.concepts)


class _ConceptState:
    def __init__(self, concepts):
        self.concepts = concepts
        self.residual = None
        self.concept_out = {}
        self.image_out = {}
        self.concept_pe = None
        self.cache = set()


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


def _encode_concepts(clip, concepts, strip_prefix):
    """Encode each concept and average its token span into a single embedding row."""
    vectors = []
    for concept in concepts:
        context, token_ids = _encode_context(clip, concept)
        start, end = _concept_span(token_ids)
        if strip_prefix:
            lo, hi = 0, end - start
        else:
            lo, hi = start, end
        if hi <= lo:
            lo, hi = 0, 1
        span = context[:, lo:hi]
        vectors.append(span.mean(dim=1).squeeze(0))
    return torch.stack(vectors, dim=0).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Entry point
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


def compute_concept_attention(model, vae, clip, image, prompt, concepts,
                              layer_start=None, layer_end=None, num_steps=4,
                              noise_timestep=2, seed=0, softmax=True,
                              temperature=1.0):
    base = model.model
    dit = base.diffusion_model
    is_krea2 = isinstance(base, comfy.model_base.Krea2)
    if not is_krea2 and not getattr(getattr(dit, "params", None), "global_modulation", False):
        raise NotImplementedError("ConceptAttention supports Flux.2 / Flux.2 Klein and Krea 2 models")

    concepts = [c.strip() for c in concepts if c and c.strip()]
    if not concepts:
        raise ValueError("concept_list must contain at least one concept")

    layer_indices = _layer_indices_for(dit, layer_start, layer_end)

    # --- encode image to latent, add noise at the requested timestep ---
    latent = vae.encode(image[..., :3])
    latent = base.process_latent_in(latent)

    model_sampling = base.model_sampling
    t_value = float(model_sampling.percent_to_sigma(min(max(noise_timestep / max(num_steps, 1), 0.0), 0.999)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(latent.device).to(latent.dtype)
    noisy = (1.0 - t_value) * latent + t_value * noise
    t = torch.full((latent.shape[0],), t_value, device=latent.device, dtype=latent.dtype)

    # --- text conditioning and concept embeddings ---
    context, _ = _encode_context(clip, prompt)
    concept_ctx = _encode_concepts(clip, concepts, strip_prefix=is_krea2)

    comfy.model_management.load_model_gpu(model)
    dtype = base.get_dtype_inference()
    device = getattr(model, "load_device", None) or comfy.model_management.get_torch_device()
    latent = latent.to(device)
    noisy = noisy.to(device)
    concept_ctx = comfy.model_management.cast_to_device(concept_ctx, device, dtype)

    state = _ConceptState(concepts)
    state.cache = set(layer_indices)
    if is_krea2:
        _prepare_krea2(dit, state, concept_ctx)
    else:
        _prepare_flux2(dit, state, concept_ctx)
    grid = _grid_shape(dit, latent)

    transformer_options = {}
    if "transformer_options" in model.model_options:
        transformer_options = pe.merge_nested_dicts(transformer_options, model.model_options["transformer_options"], copy_dict1=False)
    transformer_options["concept_state"] = state

    if is_krea2:
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "concept_attention", _krea2_wrapper, transformer_options)
    else:
        _install_flux2_patches(dit, state, transformer_options)

    base.apply_model(noisy, t, c_crossattn=context, transformer_options=transformer_options)

    return _build_maps(state, layer_indices, softmax, temperature, image, latent, grid)


# ---------------------------------------------------------------------------
# Flux.2 / Flux.2 Klein
# ---------------------------------------------------------------------------

def _prepare_flux2(dit, state, concept_ctx):
    state.residual = dit.txt_in(concept_ctx)
    axes = len(dit.params.axes_dim)
    concept_ids = torch.zeros((1, state.residual.shape[1], axes), device=state.residual.device, dtype=torch.float32)
    state.concept_pe = dit.pe_embedder(concept_ids)


def _install_flux2_patches(dit, state, transformer_options):
    patches = {("double_block", i): _flux2_block_replacement(block, state, i)
               for i, block in enumerate(dit.double_blocks)}
    transformer_options["patches_replace"] = {"dit": patches}


def _mod_pair(vec, block):
    if isinstance(vec, tuple):
        return vec[0], vec[1]
    return block.img_mod(vec), block.txt_mod(vec)


def _flux2_block_replacement(block, state, layer_index):
    """Double-block wrapper that also runs the concept residual stream."""
    def replacement(args, extra):
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
        concept_attn = flux_attention(cq, ck, cv, pe=concept_pe)[:, :state.residual.shape[1]]

        if layer_index in state.cache:
            state.concept_out[layer_index] = concept_attn[0].detach().float().cpu()
            state.image_out[layer_index] = img_attn[0].detach().float().cpu()

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

def _prepare_krea2(dit, state, concept_ctx):
    num_concepts = concept_ctx.shape[1]
    packed = concept_ctx.reshape(concept_ctx.shape[0], num_concepts, dit.txtlayers, dit.txtdim)
    fused = dit.txtfusion(packed, mask=None, transformer_options={})
    state.residual = dit.txtmlp(fused)
    concept_pos = torch.zeros((1, num_concepts, 3), device=state.residual.device, dtype=torch.float32)
    state.concept_pe = dit.pe_embedder(concept_pos)


def _krea2_wrapper(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
    dit = executor.class_obj
    state = transformer_options["concept_state"]
    _krea2_concept_forward(dit, x, timesteps, context, state)
    return torch.zeros_like(x)


def _repeat_kv(t, heads):
    if t.shape[1] == heads:
        return t
    return t.repeat_interleave(heads // t.shape[1], dim=1)


def _krea2_concept_forward(dit, x, timesteps, context, state):
    bs = x.shape[0]
    context = dit._unpack_context(context)
    img, imgpos, _, _ = dit.process_img(x)
    img = dit.first(img)

    t = dit.tmlp(timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(img.dtype))
    tvec = dit.tproj(t)

    context = dit.txtfusion(context, mask=None, transformer_options={})
    context = dit.txtmlp(context)

    txt_len = context.shape[1]
    txtpos = torch.zeros(bs, txt_len, 3, device=context.device, dtype=torch.float32)
    img_tokens = img.shape[1]
    combined = torch.cat((context, img), dim=1)
    del context, img

    freqs = dit.pe_embedder(torch.cat((txtpos, imgpos), dim=1))

    concepts = state.residual
    for index, block in enumerate(dit.blocks):
        combined, concepts = _krea2_block(block, combined, concepts, tvec, freqs,
                                          state.concept_pe, txt_len, img_tokens, state, index)
    state.residual = concepts


def _krea2_block(block, combined, concepts, tvec, freqs, concept_freqs,
                 txt_len, img_tokens, state, layer_index):
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(tvec)
    attn = block.attn
    heads, kvheads = attn.heads, attn.kvheads
    head_dim = attn.headdim

    pre = (1 + prescale) * block.prenorm(combined) + preshift
    q = attn.wq(pre).view(pre.shape[0], pre.shape[1], heads, head_dim).transpose(1, 2)
    k = attn.wk(pre).view(pre.shape[0], pre.shape[1], kvheads, head_dim).transpose(1, 2)
    v = attn.wv(pre).view(pre.shape[0], pre.shape[1], kvheads, head_dim).transpose(1, 2)
    gate = attn.gate(pre)
    q, k = attn.qknorm(q, k)

    q, k = apply_rope(q, k, freqs)
    out = optimized_attention_masked(q, _repeat_kv(k, heads), _repeat_kv(v, heads), heads, skip_reshape=True)
    img_attn = out[:, txt_len:txt_len + img_tokens]

    cpre = (1 + prescale) * block.prenorm(concepts) + preshift
    cq = attn.wq(cpre).view(cpre.shape[0], cpre.shape[1], heads, head_dim).transpose(1, 2)
    ck = attn.wk(cpre).view(cpre.shape[0], cpre.shape[1], kvheads, head_dim).transpose(1, 2)
    cv = attn.wv(cpre).view(cpre.shape[0], cpre.shape[1], kvheads, head_dim).transpose(1, 2)
    cgate = attn.gate(cpre)
    cq, ck = attn.qknorm(cq, ck)

    cq, ck = apply_rope(cq, ck, concept_freqs)
    ck = torch.cat((ck, k[:, :, txt_len:txt_len + img_tokens]), dim=2)
    cv = torch.cat((cv, v[:, :, txt_len:txt_len + img_tokens]), dim=2)
    concept_attn = optimized_attention_masked(cq, _repeat_kv(ck, heads), _repeat_kv(cv, heads), heads, skip_reshape=True)

    if layer_index in state.cache:
        state.concept_out[layer_index] = concept_attn[0].detach().float().cpu()
        state.image_out[layer_index] = img_attn[0].detach().float().cpu()

    combined = combined + pregate * attn.wo(out * torch.sigmoid(gate))
    combined = combined + postgate * block.mlp((1 + postscale) * block.postnorm(combined) + postshift)

    concepts = concepts + pregate * attn.wo(concept_attn * torch.sigmoid(cgate))
    concepts = concepts + postgate * block.mlp((1 + postscale) * block.postnorm(concepts) + postshift)
    return combined, concepts


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def _build_maps(state, layer_indices, softmax, temperature, image, latent, grid):
    scores = None
    used = 0
    for index in layer_indices:
        if index not in state.concept_out:
            continue
        dot = state.concept_out[index] @ state.image_out[index].transpose(0, 1)
        scores = dot if scores is None else scores + dot
        used += 1
    if scores is None:
        raise RuntimeError("ConceptAttention did not collect any layer outputs")

    scores = scores / used
    if softmax:
        scores = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)

    height, width = latent.shape[-2], latent.shape[-1]
    pos = state.image_out[layer_indices[0]].shape[0]
    grid_h, grid_w = grid
    if grid_h * grid_w != pos:
        raise RuntimeError(f"ConceptAttention image tokens ({pos}) do not match latent grid {grid_h}x{grid_w}")

    maps = scores.reshape(scores.shape[0], grid_h, grid_w).unsqueeze(1)
    maps = torch.nn.functional.interpolate(maps, size=(height, width), mode="bilinear", align_corners=False)
    out_h, out_w = image.shape[1], image.shape[2]
    maps = torch.nn.functional.interpolate(maps, size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(1)

    flat = maps.reshape(maps.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    maps = ((flat - lo) / (hi - lo + 1e-8)).reshape(maps.shape).float()

    return ConceptMaps(list(state.concepts), maps, out_h, out_w)
