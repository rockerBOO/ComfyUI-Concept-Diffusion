# ComfyUI-Concept-Diffusion

ComfyUI nodes for [ConceptAttention: Diffusion Transformers Learn Highly
Interpretable Features](https://arxiv.org/abs/2502.04320).

ConceptAttention produces sharp per-concept saliency maps from the *attention
output space* of a multi-modal diffusion transformer. It needs no training: each
concept is embedded as a token, run through a parallel residual stream that
reuses the model's own text attention weights, and scored against the image
patch attention outputs by a linear projection.

This is a real implementation. It runs the concept stream with the model's own
(possibly quantized) modules, so **nvfp4 and int8-convrot** checkpoints work
through the normal ComfyUI ops.

## Supported models

| Family | ComfyUI class | Text encoder | VAE |
|--------|---------------|--------------|-----|
| **Krea 2** | `comfy.ldm.krea2.SingleStreamDiT` | Qwen3-VL-4B (`krea2`) | Wan 2.1 |
| **Flux.2 Klein / Flux.2** | `comfy.ldm.flux.Flux` (`global_modulation`) | Qwen3-4B/8B (`flux2`) or Mistral3 (`flux2`) | Flux.2 |

Krea 2 is a single-stream MMDiT, so the concept attention is adapted to it:
concept queries attend to `[concept keys, image keys]` while the image output
comes from the normal `[text + image]` attention. Flux.2 uses the paper's
double-stream formulation directly.

## Nodes

### Concept Attention
Give it a `MODEL`, `VAE`, `CLIP`, an `IMAGE`, a `prompt`, and a comma separated
`concepts` list. It encodes the image, adds noise at `noise_timestep` (of
`num_steps`), runs one forward pass, and returns:

- `concept_maps` – `CONCEPT_MAPS`, normalized `(C, H, W)` maps
- `heatmaps` – an `IMAGE` row of per-concept plasma heatmaps with labels
- `overlay` – the input image with all concept maps tinted and blended

`layer_start` / `layer_end` select the transformer blocks to average over (`-1`
= the last 4 double/single blocks). `softmax` (on by default) normalizes across
concepts per pixel. `temperature` divides the scores before softmax.

### Concept Attention Visualizer
Takes `concept_maps` and an `IMAGE`. Set `concept_name` to a single concept to
overlay just that map, or leave it empty to tint all concepts.

### Concept Saliency Map
Takes `concept_maps` and thresholds one concept into a `MASK` plus a colored
saliency `IMAGE`.

## Usage

1. Load an image, a Krea 2 or Flux.2 (Klein) diffusion model, its text encoder,
   and its VAE.
2. Connect them to **Concept Attention** with a prompt and a concept list.
3. Save the `heatmaps` and/or `overlay` outputs.

See `example_workflow.json` for a Krea 2 graph. To use Flux.2 Klein, switch the
`UNETLoader` to a Klein model, set the `CLIPLoader` type to `flux2` with
`qwen_3_4b` (4B) or `qwen_3_8b` (9B), and point `VAELoader` at the Flux.2 VAE.

## Notes

- The paper's `encode_image` defaults are `noise_timestep=2`, `num_steps=4`.
- Increase `num_steps`/`noise_timestep` for noisier attribution.

## Based on

[ConceptAttention](https://github.com/helblazer811/ConceptAttention) ·
[arXiv:2502.04320](https://arxiv.org/abs/2502.04320)

## License

MIT
