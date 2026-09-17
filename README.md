# ComfyUI-Concept-Diffusion

ComfyUI nodes for [ConceptAttention: Diffusion Transformers Learn Highly
Interpretable Features](https://arxiv.org/abs/2502.04320).

ConceptAttention produces sharp per-concept saliency maps from the *attention
output space* of a multi-modal diffusion transformer. It needs no training: each
concept is embedded as a token, run through a parallel residual stream that
reuses the model's own text attention weights, and scored against the image
patch attention outputs by a linear projection.

Capture is observational — it never changes what the model produces — and it
runs through the model's own (possibly quantized) modules, so **nvfp4 and
int8-convrot** checkpoints work.

## Supported models

| Family | ComfyUI class | Text encoder | VAE |
|--------|---------------|--------------|-----|
| **Krea 2** | `comfy.ldm.krea2.SingleStreamDiT` | Qwen3-VL-4B (`krea2`) | Wan 2.1 |
| **Flux.2 Klein / Flux.2** | `comfy.ldm.flux.Flux` (`global_modulation`) | Qwen3-4B/8B (`flux2`) or Mistral3 (`flux2`) | Flux.2 |

Krea 2 is a single-stream MMDiT, so concept queries attend to
`[concept keys, image keys]` while the image output comes from the normal
`[text + image]` attention. Flux.2 uses the paper's double-stream formulation.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI. Only
`torch`, `numpy`, `Pillow` and `matplotlib` are used (all already present in a
normal ComfyUI install).

## Two modes

### Generate (the paper's main result)

`Concept Attention Model` patches a `MODEL`. Sample normally with `KSampler`,
then `Concept Attention Maps` turns the attention collected across every
denoising step into heatmaps for the generated image.

```
UNETLoader ─► LoraLoader ─┐
                          ├─ Concept Attention Model ─ model ─► KSampler ─► VAEDecode ─┐
CLIPLoader ───────────────┘            │ concept_state                                  │
                                       └──────► Concept Attention Maps ◄────────────────┘
                                                        │ heatmaps / overlay
```

### Encode (attribute an existing image)

`Concept Attention (encode image)` does it in one node: VAE-encode the image,
add noise at `noise_timestep` of `num_steps`, run one forward pass, and return
the maps plus an overlay.

## Nodes

- **Concept Attention (encode image)** — image in, `concept_maps` + labeled
  `heatmaps` + `overlay` out.
- **Concept Attention Model (generate)** — `MODEL` + `CLIP` + concept list in,
  patched `MODEL` + `concept_state` out.
- **Concept Attention Maps** — `concept_state` + decoded `IMAGE` in, maps out.
- **Concept Attention Visualizer** — overlay one concept (or all) onto an image.
- **Concept Saliency Map** — threshold one concept into a `MASK` + saliency image.

## Settings

- `temperature` — defaults to **1000**, the paper's Flux.2 value. Lower it
  toward `1` for sharper, more binary maps; raise it for smoother maps.
- `softmax` — normalizes across concepts per pixel (keep on).
- `layer_start` / `layer_end` — which transformer blocks to average over.
  `-1` = the last 4 blocks.
- `noise_timestep` / `num_steps` — encode mode only; the paper uses `2` and `4`.

## Example workflows

In [`example_workflows/`](example_workflows):

| File | Description |
|------|-------------|
| `generate_krea2.json` | Generate + capture, Krea 2 (UI format) |
| `generate_krea2_api.json` | Same graph in API format (`POST /prompt`) |
| `encode_image.json` | Attribute an existing image (UI format) |

## Gotchas

- **LoRA ordering.** A LoRA must come *before* `Concept Attention Model`, and
  `Concept Attention Model` must be the **last model node** into `KSampler`. If
  they are wired in parallel, the sampler runs the unpatched model and nothing is
  collected.
- **Concepts must appear in the prompt.** A concept that isn't in the prompt
  (e.g. `dragon` on a photo of a woman) produces a diffuse, unrelated map. That's
  expected.
- **Fresh per run.** `Concept Attention Model` is marked non-cacheable, so each
  run collects into a new state. Reusing one `concept_state` across two samplers
  in the same graph would mix them.
- **Map resolution** is the model's latent token grid (e.g. 64×64 for Krea 2 at
  1024), upsampled to the image. `temperature` controls softness, not resolution.
- When CFG is active, only the positive/cond batch is scored.

## Based on

[ConceptAttention](https://github.com/helblazer811/ConceptAttention) ·
[arXiv:2502.04320](https://arxiv.org/abs/2502.04320)

## License

MIT — see [LICENSE](LICENSE). Original work © 2025 Junst; rewrite © 2026 rockerBOO.
