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

## Two modes

### Generate (the paper's main result)

`Concept Attention Model` patches a MODEL. Sample normally with `KSampler`, then
`Concept Attention Maps` turns the attention collected across every denoising
step into heatmaps for the generated image.

```
UNETLoader ─┐
            ├─ Concept Attention Model ── model ──► KSampler ──► VAEDecode ──┐
CLIPLoader ─┘            │ concept_state                                     │
                         └────────────────► Concept Attention Maps ◄─────────┘
                                                    │ heatmaps / overlay
```

See `example_workflow.json`.

### Encode (attribute an existing image)

`Concept Attention (encode image)` does it in one node: VAE-encode the image,
add noise at `noise_timestep` of `num_steps`, run one forward pass, and return
the maps plus an overlay.

See `example_workflow_encode.json`.

## Nodes

- **Concept Attention (encode image)** — image in, `concept_maps` +
  labeled `heatmaps` + `overlay` out.
- **Concept Attention Model (generate)** — MODEL + CLIP + concept list in,
  patched MODEL + `concept_state` out.
- **Concept Attention Maps** — `concept_state` + decoded IMAGE in, maps out.
- **Concept Attention Visualizer** — overlay one concept (or all) onto an image.
- **Concept Saliency Map** — threshold one concept into a `MASK` + saliency image.

`layer_start` / `layer_end` select the transformer blocks to average over (`-1`
= the last 4). `softmax` (on by default) normalizes across concepts per pixel;
`temperature` divides the scores first.

## Notes

- The paper's `encode_image` defaults are `noise_timestep=2`, `num_steps=4`.
- Only the positive/cond batch is scored when CFG is active.
- Raise `temperature` if maps look washed out, lower it if too binary.

## Based on

[ConceptAttention](https://github.com/helblazer811/ConceptAttention) ·
[arXiv:2502.04320](https://arxiv.org/abs/2502.04320)

## License

MIT
