# MiniMax H3 Studio

A local, standalone browser harness for MiniMax H3 Image GGUF generation via `stable-diffusion.cpp` (`sd-cli.exe`). No ComfyUI, bundled weights, or hosted inference.

## Windows setup

1. Install Python 3.10+ and get a MiniMax H3-compatible CUDA build of `sd-cli.exe` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
2. Download one MiniMax H3 Image GGUF diffusion model, a compatible text encoder and the MiniMax H3 Image VAE.
3. Run `START_WINDOWS.bat`. The local interface opens at `http://127.0.0.1:7865`.
4. Use **Browse** to select the local executable, diffusion model, text encoder and VAE. Enter a prompt, then generate.

## Capabilities

- Native Windows file selection on the server's own desktop
- Text-to-image and optional reference-image editing
- Optional experimental LoRA selection; model compatibility is **not yet confirmed**
- Local-only web interface, saved gallery, logs, and persistent file selections
- Memory-conscious defaults for NVIDIA RTX A4500 20 GB

**Security:** Local host only. Do not publish this local app to the internet. Downloads and models are not included in this repo. File selectors operate on the computer hosting the app.

**Status:** Source scaffold only; model loading and CLI flags have not been runtime-verified on the owner's Windows machine.

Docs: [MiniMax model guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/models/minimax_h3_image.md) · [LoRA guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/lora.md)
