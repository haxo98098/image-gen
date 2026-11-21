# Qwen Image Studio

A local browser harness for **Qwen-Image 2.1** using `stable-diffusion.cpp` (`sd-cli.exe`). No ComfyUI and no hosted inference.

## Recommended files for this machine

For an RTX A4500 with 20 GB VRAM, start with:

- Diffusion model: `qwen-image-2.1-UC-Q4_K_M.gguf` (~4.60 GB)
- Text encoder: `qwen3vl_8b_int8_convrot.safetensors` (~9.35 GB)
- VAE: `qwen_image_2.1_vae_bf16.safetensors` (~676 MB)
- A current CUDA build of `sd-cli.exe` from stable-diffusion.cpp

Qwen-Image 2.1 requires its own VAE; older Qwen Image / Wan VAE files are not interchangeable.

## Simple folder layout

Put the files together if you want automatic discovery:

```text
Qwen-Image/
├── qwen-image-2.1-UC-Q4_K_M.gguf
├── qwen3vl_8b_int8_convrot.safetensors
├── qwen_image_2.1_vae_bf16.safetensors
├── sd-cli.exe
└── any DLLs shipped with sd-cli.exe
```

The app also checks common `text_encoders/`, `vae/`, `models/`, and `bin/` subfolders.

## Windows setup

1. Install Python 3.10+.
2. Put the model, encoder, VAE, and current `sd-cli.exe` build in one folder (or nearby recognized subfolders).
3. Run `START_WINDOWS.bat`.
4. Click **Select Qwen-Image 2.1 Model** and choose the GGUF.
5. Studio searches for the encoder, VAE, and `sd-cli.exe`. Browse buttons appear only for anything it cannot find.
6. Enter a prompt and generate.

## Generation defaults

The harness follows the stable-diffusion.cpp Qwen-Image 2.1 path:

- `--diffusion-model`
- `--llm`
- `--vae`
- Euler sampler
- CFG 6.0
- 20 steps
- CPU offload enabled
- VAE tiling enabled
- image dimensions must be divisible by 32

## Capabilities

- One main model picker
- Automatic companion-file discovery
- Native Windows Browse dialogs only when something is missing
- Text-to-image generation
- Optional reference-image editing
- Optional experimental LoRA selection
- Local gallery and generation logs
- Local-only HTTP server on `127.0.0.1:7865`

**Security:** This is intended as a local application. Do not expose the Flask/Waitress server directly to the public internet.

**Status:** Source changes are committed but have not been runtime-tested on the owner's Windows machine.

Docs: [stable-diffusion.cpp Qwen Image 2.1 guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md)
