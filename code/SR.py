#pip install git+https://github.com/huggingface/diffusers.git

import requests
import torchvision
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# let's download an  image
url = "https://user-images.githubusercontent.com/38061659/199705896-b48e17b8-b231-47cd-a270-4ffa5a93fa3e.png"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
# save image
upscaled_image.save("ldm_generated_image.png")


from diffusers import DiffusionPipeline
import os

# 모델 ID 및 로컬 저장 경로 설정
model_id = "CompVis/ldm-super-resolution-4x-openimages"
local_model_path = "path/to/local/model/directory"

# 모델 다운로드 또는 로드
pipeline = DiffusionPipeline.from_pretrained(model_id)

# 로컬에 저장
pipeline.save_pretrained(local_model_path)
