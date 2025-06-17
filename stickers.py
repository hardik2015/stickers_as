import sys
from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.loaders import AttnProcsLayers
from PIL import Image
import io
import requests
from requests.auth import HTTPBasicAuth
import uuid

if len(sys.argv) < 5:
    print("parameters passed as less .")

app = Flask(__name__)
# ==== Load SDXL base model + LoRA ====
print("🔧 Loading model...")
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "./sticker-lora-dev.safetensors"  # Update with your LoRA path

if(sys.argv[4] == "gpu"):
  pipe = StableDiffusionXLPipeline.from_pretrained(
      base_model,
      torch_dtype=torch.float16,
      variant="fp16",
      use_safetensors=True
  ).to("cuda")
else:
  pipe = StableDiffusionXLPipeline.from_pretrained(
      base_model,
      torch_dtype=torch.float16,
      variant="fp16",
      use_safetensors=True
  )

pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
print("✅ Model ready.")

# ==== WebDAV (ownCloud/Nextcloud) ====
webdav_url_base = sys.argv[1]
webdav_user = sys.argv[2]
webdav_pass = sys.argv[3]

@app.route("/generate", methods=["POST"])
def generate_images():
    data = request.get_json()
    prompt_string = data.get("prompt")

    if not prompt_string:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    # Split comma-separated words
    words = [word.strip() for word in prompt_string.split(",") if word.strip()]
    if not words:
        return jsonify({"error": "No valid words provided"}), 400

    image_urls = []

    for word in words:
        try:
            # Create prompt
            full_prompt = f"{word}, coloring book style, line art"
            image = pipe(prompt=full_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

            # Save image to bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
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

            if response.status_code in [200, 201, 204]:
                image_urls.append(upload_url)
            else:
                image_urls.append({"word": word, "error": f"Failed to upload (HTTP {response})"})

        except Exception as e:
            image_urls.append({"word": word, "error": str(e)})

    return jsonify({"images": image_urls})

if(__name__ == "__main__"):
    app.run(host="0.0.0.0",port=5000)
