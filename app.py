"""Local-only MiniMax H3 frontend for stable-diffusion.cpp."""
import json
import os
import secrets
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CONFIG = ROOT / "local-config.json"
app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 1_000_000
LOCK = threading.Lock()
STATE = {"running": False, "log": [], "error": None, "output": None, "returncode": None}
FIELDS = ("cli", "model", "encoder", "vae", "reference", "lora")


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
        "reference": (".png", ".jpg", ".jpeg", ".webp"),
    }
    if path.suffix.lower() not in allowed[field]:
        raise ValueError(f"Unexpected {field} file type")
    return path


def pick_file(field):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        types = ([("Executables", "*.exe")] if field == "cli"
                 else [("Images", "*.png *.jpg *.jpeg *.webp")] if field == "reference"
                 else [("Model files", "*.gguf *.safetensors")])
        return filedialog.askopenfilename(title="Select " + field, filetypes=types + [("All files", "*.*")], parent=root)
    finally:
        root.destroy()


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/config")
def get_config():
    return jsonify(config_read())


@app.post("/api/pick/<field>")
def pick(field):
    if field not in FIELDS:
        return jsonify(error="Unknown file field"), 400
    try:
        selected = pick_file(field)
        if selected:
            local_path(selected, field, required=True)
            cfg = config_read()
            cfg[field] = selected
            config_write(cfg)
        return jsonify(path=selected, config=config_read())
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


def run_generator(cmd, output):
    try:
        with subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ) as proc:
            for line in proc.stdout:
                with LOCK:
                    STATE["log"].append(line.rstrip()[:1000])
                    STATE["log"] = STATE["log"][-150:]
            code = proc.wait()
        with LOCK:
            STATE["returncode"] = code
            if code == 0 and output.is_file():
                STATE["output"] = output.name
            else:
                STATE["error"] = "Generator failed or produced no PNG; inspect the log."
    except Exception as exc:
        with LOCK:
            STATE["error"] = str(exc)
    finally:
        with LOCK:
            STATE["running"] = False


def build_command(cfg, data, output):
    cli = local_path(cfg.get("cli"), "cli", True)
    model = local_path(cfg.get("model"), "model", True)
    encoder = local_path(cfg.get("encoder"), "encoder", True)
    vae = local_path(cfg.get("vae"), "vae", True)
    prompt = str(data.get("prompt", "")).strip()
    if not prompt or len(prompt) > 6000:
        raise ValueError("Enter a prompt up to 6000 characters")
    width, height = int(data.get("width", 1024)), int(data.get("height", 1024))
    if any(n < 256 or n > 2048 or n % 32 for n in (width, height)):
        raise ValueError("Width and height must be multiples of 32, from 256 to 2048")
    seed = int(data.get("seed", -1))
    if not -1 <= seed <= 2147483647:
        raise ValueError("Seed is out of range")
    cmd = [
        str(cli), "--diffusion-model", str(model), "--llm", str(encoder),
        "--vae", str(vae), "-p", prompt, "-W", str(width), "-H", str(height),
        "--seed", str(seed), "--steps", "6", "--cfg-scale", "1.0",
        "--sampling-method", "euler", "--sigmas", "minimax-h3-image",
        "--rng", "cpu", "--offload-to-cpu", "--vae-tiling",
        "-o", str(output),
    ]
    if data.get("flash_attention", True):
        cmd.append("--diffusion-fa")
    if data.get("use_reference"):
        cmd += ["-r", str(local_path(cfg.get("reference"), "reference", True))]
    if data.get("use_lora"):
        lora = local_path(cfg.get("lora"), "lora", True)
        strength = float(data.get("lora_strength", 1.0))
        if not 0 <= strength <= 2:
            raise ValueError("LoRA strength must be between 0 and 2")
        cmd[cmd.index("-p") + 1] += f" <lora:{lora.stem}:{strength:g}>"
        cmd += ["--lora-model-dir", str(lora.parent)]
    return cmd


@app.post("/api/generate")
def generate():
    with LOCK:
        if STATE["running"]:
            return jsonify(error="A generation is already running"), 409
        try:
            data = request.get_json(silent=True) or {}
            name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3) + ".png"
            output = OUT / name
            cmd = build_command(config_read(), data, output)
        except (ValueError, TypeError, OSError) as exc:
            return jsonify(error=str(exc)), 400
        STATE.update(running=True, log=[], error=None, output=None, returncode=None)
        threading.Thread(target=run_generator, args=(cmd, output), daemon=True).start()
    return jsonify(started=True)


if __name__ == "__main__":
    from waitress import serve
    print("MiniMax H3 Studio: http://127.0.0.1:7865")
    serve(app, host="127.0.0.1", port=7865, threads=4)
