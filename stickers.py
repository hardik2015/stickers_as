from flask import Flask, request, jsonify
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.loaders import AttnProcsLayers
from PIL import Image
import threading
import uuid
import time
import torch
import io
import requests
from requests.auth import HTTPBasicAuth
import gc
import sys
from rembg import remove


if len(sys.argv) < 5:
    print("parameters passed as less .")
    
app = Flask(__name__)
TASKS = {}
print("🔧 Loading model...")
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "./StickersRedmond.safetensors"  # Update with your LoRA path

pipe = StableDiffusionXLPipeline.from_pretrained(
  base_model,
  torch_dtype=torch.float16,
  variant="fp16",
  use_safetensors=True
)

pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
if(sys.argv[4] == "gpu"):
    pipe.to("cuda")

print("✅ Model ready.")

# ==== WebDAV (ownCloud/Nextcloud) ====
webdav_url_base = sys.argv[1]
webdav_user = sys.argv[2]
webdav_pass = sys.argv[3]

def generate_image_task(prompt, task_id):
    try:
        words = [word.strip() for word in prompt.split(",") if word.strip()]
        if not words:
          TASKS[task_id] = {"status": "error", "message": str(e)}
          return

        count = 0
        TASKS[task_id] = {"status": "done", "count": count}
        # simulate long generation
        for word in words:
          try:
              for i in range(2):  # Iterates from 0 to 4
                # Create prompt
                full_prompt = f"{word}, stickers, simple, <lora:StickersRedmond:1> "
                negative_prompt = "ugly, disfigured, duplicate, mutated, bad art, blur, blurry, dof, background, multiple objects, two object, incomplete, unfinished"
                image = pipe(prompt=full_prompt, negative_prompt=negative_prompt, num_inference_steps=35, guidance_scale=7, strength=1).images[0]
                output = remove(image)
                # Save image to bytes
                img_buffer = io.BytesIO()
                output.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Upload to WebDAV
                filename = f"{uuid.uuid4().hex}_{word}.png"
                upload_url = webdav_url_base + filename

                response = requests.put(
                    upload_url,
                    data=img_bytes,
                    auth=HTTPBasicAuth(webdav_user, webdav_pass),
                    headers={"Content-Type": "image/png"}
                )
                # Helps clear memory between iterations
                torch.cuda.empty_cache()
                gc.collect()

              count = count+1
              TASKS[task_id] = {"status": "done", "count": count}

          except Exception as e:
              TASKS[task_id] = {"status": "error", "message": str(e), "count":"0"}
              print(f"[{task_id}] Error during generation: {str(e)}")
          finally:
              # Helps clear memory between iterations
              torch.cuda.empty_cache()
              gc.collect()


        TASKS[task_id] = {"status": "done", "count": "all generation done"}
        print(f"[{task_id}] Image done.")
    except Exception as e:
        TASKS[task_id] = {"status": "error", "message": str(e)}
        print(f"[{task_id}] Error during generation: {str(e)}")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "processing"}

    thread = threading.Thread(target=generate_image_task, args=(prompt, task_id), daemon=True)
    thread.start()

    return jsonify({
        "message": "Image generation started",
        "task_id": task_id
    }), 202

@app.route("/status/<task_id>")
def check_status(task_id):
    status = TASKS.get(task_id)
    if not status:
        return jsonify({"error": "Invalid task ID"}), 404
    return jsonify(status)

if(__name__ == "__main__"):
    app.run(host="0.0.0.0",port=5000)
