"""Local-only Qwen-Image 2.1 frontend for stable-diffusion.cpp."""
import json
import re
import os
import secrets
import shutil
import subprocess
import threading
import base64
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
REFS = ROOT / "references"
REFS.mkdir(exist_ok=True)
CONFIG = ROOT / "local-config.json"
app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
LOCK = threading.Lock()
CURRENT_PROCESS = None
JOB_QUEUE = []
QUEUE_WORKER_ACTIVE = False
STATE = {"running": False, "log": [], "error": None, "output": None, "returncode": None, "progress": None, "stage": "Ready", "cancel_requested": False, "queue_count": 0, "current_prompt": ""}
FIELDS = ("cli", "engine_dir", "model", "encoder", "vae", "lora", "upscaler")


def config_read():
    try:
        data = json.loads(CONFIG.read_text("utf-8"))
        return {key: str(data.get(key, "")) for key in FIELDS}
    except (OSError, ValueError, TypeError):
        return {key: "" for key in FIELDS}


def config_write(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def local_path(value, field, required=False):
    if not value:
        if required:
            raise ValueError(f"Select a {field} file first")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{field} file not found: {path}")
    allowed = {
        "cli": (".exe",) if os.name == "nt" else ("", ".exe"),
        "model": (".gguf", ".safetensors"),
        "encoder": (".gguf", ".safetensors"),
        "vae": (".gguf", ".safetensors"),
        "lora": (".gguf", ".safetensors"),
        "upscaler": (".pth", ".pt", ".safetensors", ".gguf"),
    }
    if path.suffix.lower() not in allowed[field]:
        raise ValueError(f"Unexpected {field} file type")
    if field == "model" and not is_qwen_image_model(path):
        lower = path.name.lower()
        if any(word in lower for word in ("qwen3vl", "text_encoder", "text-encoder")):
            raise ValueError("That file is the Qwen text encoder, not the image diffusion model.")
        if "vae" in lower or "autoencoder" in lower:
            raise ValueError("That file is the VAE, not the image diffusion model.")
    return path


def pick_file(field):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if field == "engine_dir":
            return filedialog.askdirectory(
                title="Select extracted stable-diffusion.cpp folder",
                mustexist=True,
                parent=root,
            )
        if field == "cli":
            types = [("Executables", "*.exe")]
        elif field == "upscaler":
            types = [("Upscaler models", "*.pth *.pt *.safetensors *.gguf")]
        else:
            types = [("Model files", "*.gguf *.safetensors")]
        return filedialog.askopenfilename(
            title="Select " + field,
            filetypes=types + [("All files", "*.*")],
            parent=root,
        )
    finally:
        root.destroy()


def find_sd_cli(folder):
    if not folder:
        return None
    root = Path(folder).expanduser()
    if not root.is_dir():
        return None
    preferred = (
        root / "sd-cli.exe",
        root / "sd-cli",
        root / "bin" / "Release" / "sd-cli.exe",
        root / "build" / "bin" / "Release" / "sd-cli.exe",
        root / "bin" / "sd-cli.exe",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    try:
        for name in ("sd-cli.exe", "sd-cli"):
            match = next(root.rglob(name), None)
            if match and match.is_file():
                return match.resolve()
    except OSError:
        pass
    return None



def is_qwen_image_model(path):
    if not path or not path.is_file() or path.suffix.lower() not in (".gguf", ".safetensors"):
        return False
    lower = path.name.lower()
    # Never mistake known companion components for the diffusion model.
    if any(word in lower for word in (
        "qwen3vl", "qwen3-vl", "qwen3_vl",
        "text_encoder", "text-encoder", "vae", "autoencoder"
    )):
        return False
    # GGUF files in this dedicated project folder are valid diffusion-model
    # candidates even when a community quantizer used a nonstandard filename.
    if path.suffix.lower() == ".gguf":
        return True
    # Safetensors need a stronger filename signal to avoid grabbing random files.
    return ("qwen" in lower and "image" in lower) or lower.startswith("qwen-image-2.1")


def find_qwen_image_model():
    candidates = []
    try:
        # Search the project tree recursively so users can drop the model anywhere
        # under image-gen without having to follow a special folder layout.
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if any(part in (".venv", "outputs", "__pycache__") for part in path.parts):
                continue
            if is_qwen_image_model(path):
                candidates.append(path.resolve())
    except OSError:
        pass
    if not candidates:
        return None

    def score(path):
        name = path.name.lower()
        rank = 0
        # Prefer filenames that explicitly identify Qwen Image / desired quant.
        if "qwen" in name:
            rank += 100
        if "image" in name:
            rank += 100
        for token, value in (
            ("2.1", 90), ("q4_k_m", 80), ("q5_k_m", 70), ("q6_k", 60),
            ("q8_0", 50), ("fp8", 40), ("bf16", 30),
            ("uncensored", 20), ("uc", 15),
        ):
            if token in name:
                rank += value
        if path.suffix.lower() == ".gguf":
            rank += 10
        # If names are unhelpful, prefer the larger model file over tiny utility GGUFs.
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return (-rank, -size, path.name.casefold())

    return sorted(candidates, key=score)[0]


REQUIRED = ("cli", "model", "encoder", "vae")


def discover_companions(cfg):
    """Resolve the engine and companion model files with minimal user setup."""
    cfg = dict(cfg)

    # Auto-detect the main Qwen image diffusion model from the project folder.
    if not cfg.get("model") or not Path(cfg["model"]).is_file():
        detected_model = find_qwen_image_model()
        if detected_model:
            cfg["model"] = str(detected_model)

    # Migrate old UI mistakes: an encoder/VAE selected in the former "diffusion model" field
    # should not be treated as the image model.
    if cfg.get("model"):
        old_model = Path(cfg["model"]).expanduser()
        lower = old_model.name.lower()
        if old_model.is_file() and any(word in lower for word in ("qwen3vl", "qwen3-vl", "qwen3_vl", "text_encoder", "text-encoder")):
            if not cfg.get("encoder"):
                cfg["encoder"] = str(old_model.resolve())
            cfg["model"] = ""
        elif old_model.is_file() and ("vae" in lower or "autoencoder" in lower):
            if not cfg.get("vae"):
                cfg["vae"] = str(old_model.resolve())
            cfg["model"] = ""

    # The migration above may have cleared a bad old model selection (for example,
    # the Qwen3-VL encoder). Detect the real diffusion model immediately afterward.
    if not cfg.get("model"):
        detected_model = find_qwen_image_model()
        if detected_model:
            cfg["model"] = str(detected_model)

    model = Path(cfg["model"]).expanduser() if cfg.get("model") else None

    # Engine: use the selected stable-diffusion.cpp folder, PATH, then nearby app folders.
    cli = find_sd_cli(cfg.get("engine_dir"))
    if cli:
        cfg["cli"] = str(cli)
    elif cfg.get("cli") and Path(cfg["cli"]).is_file():
        pass
    else:
        cfg["cli"] = ""
        cmd = shutil.which("sd-cli") or shutil.which("sd-cli.exe")
        if cmd:
            cfg["cli"] = str(Path(cmd).resolve())
        else:
            for folder in dict.fromkeys((
                model.parent if model and model.is_file() else ROOT,
                model.parent.parent if model and model.is_file() else ROOT,
                ROOT, ROOT / "bin", ROOT / "stable-diffusion.cpp",
                Path.home() / "Downloads",
            )):
                found = find_sd_cli(folder)
                if found:
                    cfg["cli"] = str(found)
                    break

    dirs = [ROOT, ROOT / "models"]
    if model and model.is_file():
        base = model.parent
        dirs = [
            base,
            base / "text_encoders",
            base / "vae",
            base / "models",
            base.parent,
            base.parent / "text_encoders",
            base.parent / "vae",
            base.parent / "models",
        ] + dirs

    for key in ("encoder", "vae"):
        value = cfg.get(key, "")
        if value and not Path(value).is_file():
            cfg[key] = ""

    # Search only nearby known folders; never scan the whole machine.
    for folder in dict.fromkeys(dirs):
        if not folder.is_dir():
            continue
        try:
            candidates = sorted(
                (p for p in folder.iterdir() if p.is_file()),
                key=lambda p: p.name.casefold(),
            )
        except OSError:
            continue
        for path in candidates:
            lower = path.name.lower()
            if path.suffix.lower() not in (".gguf", ".safetensors"):
                continue
            if model and path.resolve() == model.resolve():
                continue
            if not cfg.get("encoder") and any(
                word in lower
                for word in ("qwen3vl", "qwen3-vl", "qwen3_vl", "text_encoder", "text-encoder")
            ):
                cfg["encoder"] = str(path.resolve())
            if not cfg.get("vae") and (
                "qwen_image_2.1_vae" in lower
                or "qwen-image-2.1-vae" in lower
                or "qwen_image_vae" in lower
                or "vae" in lower
            ):
                cfg["vae"] = str(path.resolve())

    return cfg


def setup_payload():
    cfg = config_read()
    found = discover_companions(cfg)
    if found != cfg:
        config_write(found)
    return {"config": found, "missing": [key for key in REQUIRED if not found.get(key) or not Path(found[key]).is_file()]}


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/config")
def get_config():
    return jsonify(setup_payload())


@app.post("/api/pick/<field>")
def pick(field):
    if field not in FIELDS:
        return jsonify(error="Unknown file field"), 400
    try:
        selected = pick_file(field)
        if selected:
            cfg = config_read()
            if field == "engine_dir":
                cfg[field] = selected
                cli = find_sd_cli(selected)
                if not cli:
                    return jsonify(
                        error="No sd-cli.exe was found inside that folder. Select the extracted stable-diffusion.cpp build folder."
                    ), 400
                cfg["cli"] = str(cli)
            else:
                local_path(selected, field, required=True)
                cfg[field] = selected
            config_write(cfg)
        return jsonify(path=selected, **setup_payload())
    except Exception as exc:
        return jsonify(error=f"Could not open picker: {exc}"), 500


@app.get("/api/status")
def status():
    with LOCK:
        return jsonify(dict(STATE))


@app.get("/api/gallery")
def gallery():
    images = sorted(
        (p.name for p in OUT.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        reverse=True,
    )
    return jsonify(images=images[:100])


@app.get("/outputs/<name>")
def output_file(name):
    return send_from_directory(OUT, name)


@app.post("/api/images/<name>/reference")
def image_as_reference(name):
    try:
        source = safe_named_file(OUT, name, (".png", ".jpg", ".jpeg", ".webp"))
        target_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4) + source.suffix.lower()
        target = REFS / target_name
        shutil.copy2(source, target)
        return jsonify(name=target_name, url=f"/references/{target_name}")
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 404


@app.post("/api/cancel")
def cancel_generation():
    global CURRENT_PROCESS
    with LOCK:
        JOB_QUEUE.clear()
        STATE["queue_count"] = 0
        proc = CURRENT_PROCESS
        if not proc or proc.poll() is not None:
            return jsonify(cancelled=False, message="No generation is running")
        STATE["stage"] = "Cancelling"
        STATE["cancel_requested"] = True
        STATE["error"] = None
    try:
        proc.terminate()
        return jsonify(cancelled=True)
    except OSError as exc:
        return jsonify(error=f"Could not cancel generation: {exc}"), 500




@app.get("/references/<name>")
def reference_file(name):
    return send_from_directory(REFS, name)


def safe_named_file(folder, name, suffixes):
    clean = secure_filename(str(name or ""))
    if not clean or clean != name:
        raise ValueError("Invalid filename")
    path = (folder / clean).resolve()
    if path.parent != folder.resolve() or path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError("File not found")
    return path


@app.delete("/api/images/<name>")
def delete_image(name):
    try:
        path = safe_named_file(OUT, name, (".png", ".jpg", ".jpeg", ".webp"))
        path.unlink()
        with LOCK:
            if STATE.get("output") == name:
                STATE["output"] = None
        return jsonify(deleted=name)
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 404


@app.post("/api/references")
def upload_references():
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="Choose at least one reference image"), 400
    if len(files) > 10:
        return jsonify(error="Qwen-Image 2.1 supports up to 10 reference images"), 400
    saved = []
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    for item in files:
        original = secure_filename(item.filename or "")
        suffix = Path(original).suffix.lower()
        if suffix not in allowed:
            return jsonify(error=f"Unsupported reference image type: {suffix or 'unknown'}"), 400
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4) + suffix
        destination = REFS / name
        item.save(destination)
        saved.append({"name": name, "url": f"/references/{name}"})
    return jsonify(images=saved)


@app.delete("/api/references/<name>")
def delete_reference(name):
    try:
        path = safe_named_file(REFS, name, (".png", ".jpg", ".jpeg", ".webp"))
        path.unlink()
        return jsonify(deleted=name)
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 404


SAMPLERS = {
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep", "res_2s",
    "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge", "dpm++2m_sde",
    "dpm++2m_sde_bt", "lms"
}
SCHEDULERS = {
    "auto", "discrete", "karras", "exponential", "ays", "gits", "sgm_uniform",
    "simple", "smoothstep", "kl_optimal", "lcm", "bong_tangent", "ltx2",
    "logit_normal", "flux2", "flux", "beta", "llada_image"
}


@app.post("/api/prompt-assist")
def prompt_assist():
    data = request.get_json(silent=True) or {}
    primary = str(data.get("prompt", "")).strip()
    if not primary:
        return jsonify(error="Enter a prompt first"), 400
    if len(primary) > 6000:
        return jsonify(error="Prompt is too long"), 400

    mode = str(data.get("mode", "generate")).strip().lower()
    use_case = str(data.get("use_case", "auto")).strip().lower()
    preserve = str(data.get("preserve", "")).strip()
    avoid = str(data.get("avoid", "")).strip()
    reference_count = max(0, min(10, int(data.get("reference_count", 0) or 0)))

    lines = []
    if use_case != "auto":
        lines.append(f"Use case: {use_case}")
    lines.append(f"Primary request: {primary}")

    if reference_count:
        if mode == "edit":
            lines.append("Input images: Image 1 is the edit target" + (
                "; remaining images are supporting references" if reference_count > 1 else ""
            ))
        else:
            roles = ", ".join(f"Image {i}: reference image" for i in range(1, reference_count + 1))
            lines.append(f"Input images: {roles}")

    if mode == "edit":
        base_constraint = "Change only the requested elements; keep all unrequested parts of the edit target unchanged."
        lines.append(f"Constraints: {base_constraint}" + (f" Preserve: {preserve}" if preserve else ""))
    elif preserve:
        lines.append(f"Constraints: Preserve: {preserve}")

    if use_case == "photorealistic":
        lines.append("Style/medium: photorealistic; natural real-world texture and detail; avoid over-stylized polish unless requested.")
    elif use_case == "illustration":
        lines.append("Style/medium: illustration; keep the requested subject and composition clear.")
    elif use_case == "product":
        lines.append("Output intent: polished product image with clear silhouette, material detail, and practical composition.")
    elif use_case == "ui":
        lines.append("Output intent: practical shippable UI mockup with clear hierarchy and readable layout.")

    if avoid:
        lines.append(f"Avoid: {avoid}")
    elif mode == "generate":
        lines.append("Constraints: Do not add extra subjects, props, text, or story elements that are not implied by the request.")

    return jsonify(prompt="\n".join(lines))


def ollama_request(path, payload=None, timeout=20):
    url = "http://127.0.0.1:11434" + path
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@app.get("/api/ollama/models")
def ollama_models():
    try:
        data = ollama_request("/api/tags")
        models = []
        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models.append(name)
        return jsonify(available=True, models=models)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return jsonify(available=False, models=[], error=f"Ollama is not reachable at 127.0.0.1:11434: {exc}")


@app.post("/api/ollama/analyze")
def ollama_analyze():
    data = request.get_json(silent=True) or {}
    model = str(data.get("model", "")).strip()
    references = data.get("references") or []
    instruction = str(data.get("instruction", "")).strip()
    if not model:
        return jsonify(error="Choose an installed Ollama model"), 400
    if not references:
        return jsonify(error="Attach at least one reference image"), 400
    if len(references) > 10:
        return jsonify(error="Too many reference images"), 400
    encoded = []
    try:
        for name in references:
            path = safe_named_file(REFS, str(name), (".png", ".jpg", ".jpeg", ".webp"))
            encoded.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        user_text = instruction or "Study these reference images and write a detailed image-generation prompt that preserves their important visual traits."
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a visual prompt assistant. Inspect the supplied reference images and return only a useful, concrete image-generation prompt. Preserve requested identity, clothing, composition, lighting, style, and other visible details. Do not add commentary before or after the prompt.",
                },
                {"role": "user", "content": user_text, "images": encoded},
            ],
        }
        result = ollama_request("/api/chat", payload, timeout=180)
        text = ((result.get("message") or {}).get("content") or "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response")
        return jsonify(prompt=text)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return jsonify(error=f"Ollama vision request failed: {exc}"), 502


def run_generator(cmd, output):
    global CURRENT_PROCESS
    try:
        with subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ) as proc:
            with LOCK:
                CURRENT_PROCESS = proc
            for line in proc.stdout:
                clean = line.rstrip()
                with LOCK:
                    STATE["log"].append(clean[:1000])
                    STATE["log"] = STATE["log"][-150:]

                    # stable-diffusion.cpp prints sampling progress as "... 3/20 - 1.23s/it".
                    # This is real denoising-step progress, not a fake timer.
                    match = re.search(r"\b(\d+)\s*/\s*(\d+)\s*-\s*", clean)
                    if match:
                        completed, total = int(match.group(1)), int(match.group(2))
                        if total > 0 and 0 <= completed <= total:
                            STATE["progress"] = min(96, round((completed / total) * 96))
                            STATE["stage"] = "Creating image"
                    lower = clean.lower()
                    if any(word in lower for word in ("decode", "decoding", "vae")) and STATE.get("progress") is not None:
                        STATE["progress"] = max(STATE["progress"], 97)
                        STATE["stage"] = "Finishing image"
            code = proc.wait()
        with LOCK:
            STATE["returncode"] = code
            if code == 0 and output.is_file():
                STATE["progress"] = 100
                STATE["stage"] = "Complete"
                STATE["output"] = output.name
            elif STATE.get("cancel_requested"):
                STATE["stage"] = "Cancelled"
                STATE["error"] = None
            else:
                STATE["error"] = "Generator failed or produced no PNG; inspect the log."
    except Exception as exc:
        with LOCK:
            STATE["error"] = str(exc)
    finally:
        with LOCK:
            CURRENT_PROCESS = None
            STATE["running"] = False


def build_command(cfg, data, output):
    cli = local_path(cfg.get("cli"), "cli", True)
    model = local_path(cfg.get("model"), "model", True)
    encoder = local_path(cfg.get("encoder"), "encoder", True)
    vae = local_path(cfg.get("vae"), "vae", True)

    prompt = str(data.get("prompt", "")).strip()
    negative = str(data.get("negative_prompt", "")).strip()
    if not prompt or len(prompt) > 6000:
        raise ValueError("Enter a prompt up to 6000 characters")
    if len(negative) > 4000:
        raise ValueError("Negative prompt is too long")

    width, height = int(data.get("width", 1024)), int(data.get("height", 1024))
    if any(n < 256 or n > 2048 or n % 32 for n in (width, height)):
        raise ValueError("Width and height must be multiples of 32, from 256 to 2048")

    seed = int(data.get("seed", -1))
    if not -1 <= seed <= 2147483647:
        raise ValueError("Seed is out of range")

    steps = int(data.get("steps", 20))
    if not 1 <= steps <= 100:
        raise ValueError("Steps must be between 1 and 100")

    cfg_scale = float(data.get("cfg_scale", 6.0))
    if not 0 <= cfg_scale <= 20:
        raise ValueError("CFG scale must be between 0 and 20")

    sampler = str(data.get("sampler", "euler")).strip()
    if sampler not in SAMPLERS:
        raise ValueError("Unsupported sampling method")

    scheduler = str(data.get("scheduler", "auto")).strip()
    if scheduler not in SCHEDULERS:
        raise ValueError("Unsupported scheduler")

    use_lora = bool(data.get("use_lora"))
    if use_lora:
        lora = local_path(cfg.get("lora"), "lora", True)
        strength = float(data.get("lora_strength", 1.0))
        if not 0 <= strength <= 2:
            raise ValueError("LoRA strength must be between 0 and 2")
        prompt = prompt + f" <lora:{lora.stem}:{strength:g}>"

    cmd = [
        str(cli), "--diffusion-model", str(model), "--llm", str(encoder),
        "--vae", str(vae), "-p", prompt, "-W", str(width), "-H", str(height),
        "--seed", str(seed), "--steps", str(steps), "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampler, "--rng", "cpu", "--offload-to-cpu", "--vae-tiling",
        "-o", str(output),
    ]
    if scheduler != "auto":
        cmd += ["--scheduler", scheduler]
    if negative:
        cmd += ["--negative-prompt", negative]
    if data.get("flash_attention", True):
        cmd.append("--diffusion-fa")

    if use_lora:
        cmd += ["--lora-model-dir", str(lora.parent)]

    references = data.get("references") or []
    if len(references) > 10:
        raise ValueError("Qwen-Image 2.1 supports up to 10 reference images")
    for name in references:
        ref = safe_named_file(REFS, str(name), (".png", ".jpg", ".jpeg", ".webp"))
        cmd += ["-r", str(ref)]
    if references:
        cmd += ["--ref-image-args", "preset=qwen"]

    if data.get("use_upscaler"):
        upscaler = local_path(cfg.get("upscaler"), "upscaler", True)
        repeats = int(data.get("upscale_repeats", 1))
        if not 1 <= repeats <= 2:
            raise ValueError("Upscale repeats must be 1 or 2")
        cmd += ["--upscale-model", str(upscaler), "--upscale-repeats", str(repeats)]

    return cmd


def queue_worker():
    global QUEUE_WORKER_ACTIVE
    while True:
        with LOCK:
            if not JOB_QUEUE:
                QUEUE_WORKER_ACTIVE = False
                STATE["queue_count"] = 0
                if not STATE["running"]:
                    STATE["stage"] = "Ready"
                return
            job = JOB_QUEUE.pop(0)
            STATE.update(
                running=True,
                log=[],
                error=None,
                output=None,
                returncode=None,
                progress=None,
                stage="Preparing model",
                cancel_requested=False,
                queue_count=len(JOB_QUEUE),
                current_prompt=job["prompt"],
            )
        run_generator(job["cmd"], job["output"])


@app.post("/api/generate")
def generate():
    global QUEUE_WORKER_ACTIVE
    try:
        data = request.get_json(silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3) + ".png"
        output = OUT / name
        cmd = build_command(setup_payload()["config"], data, output)
    except (ValueError, TypeError, OSError) as exc:
        return jsonify(error=str(exc)), 400

    start_worker = False
    with LOCK:
        if len(JOB_QUEUE) >= 8:
            return jsonify(error="Generation queue is full (8 waiting jobs)"), 409
        JOB_QUEUE.append({"cmd": cmd, "output": output, "prompt": prompt})
        STATE["queue_count"] = len(JOB_QUEUE)
        if not QUEUE_WORKER_ACTIVE:
            QUEUE_WORKER_ACTIVE = True
            start_worker = True
    if start_worker:
        threading.Thread(target=queue_worker, daemon=True).start()
    return jsonify(queued=True, queue_count=STATE["queue_count"])


if __name__ == "__main__":
    from waitress import serve
    print("Qwen Image Studio: http://127.0.0.1:7865")
    serve(app, host="127.0.0.1", port=7865, threads=4)
