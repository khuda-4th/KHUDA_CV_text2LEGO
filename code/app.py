# pip install git+https://github.com/huggingface/diffusers.git

from diffusers import LDMSuperResolutionPipeline
from gradio_client import Client    ## one-2-3-45 API
from io import BytesIO
import streamlit as st
from PIL import Image
import numpy as np
import os
import torch
import time
import shutil
import trimesh
from pydeck import Deck
from io import BytesIO
from open3d import *
from open3d import visualization
import pydeck
import stable_diffusion as sd

# Streamlit UI
st.sidebar.title('Text2LEGO')
st.sidebar.header('Building LEGO 3D blueprint with prompt Using Generative AI Model  ')
st.sidebar.markdown("*using the Diffusers for Text2Image & Image Super Resolution, the One-2-3-45 for Image Segmentation and 3D Mesh conversion and ColuredVoxels2LDR for LEGO algorithm*")

st.sidebar.markdown(" ")
st.sidebar.markdown("*Step 0: 프롬프트를 바탕으로 이미지를 생성합니다.*")
st.sidebar.markdown("*Step 1: 입력한 이미지를 고해상도 이미지로 변환합니다.*")
st.sidebar.markdown("*Step 2: 변환된 고해상도 이미지에서 물체를 찾고, 그를 분할합니다.*")
st.sidebar.markdown("*Step 3: 분할된 물체를 .ply 확장자의 3d mesh로 변환합니다.*")
st.sidebar.markdown("*Step 4: 생성된 3d mesh를 LEGO 도면으로 변환합니다.*")
st.sidebar.markdown(" ")

# Image upload
st.sidebar.text("KHUDA FINAL CONFERENCE")


print(f"mps 사용 가능 여부: {torch.backends.mps.is_available()}")
print(f"mps 지원 환경 여부: {torch.backends.mps.is_built()}")

# Load super resolution model
device = torch.device('mps')
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# Check if the model is already downloaded
model_path = "../model"
local_model_path = os.path.join(model_path, model_id)

if not os.path.exists(local_model_path):
    # Download the model if it doesn't exist locally
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline.save_pretrained(local_model_path)
else:
    # Load the model from the local path
    pipeline = LDMSuperResolutionPipeline.from_pretrained(local_model_path)

# ...
st.subheader("Step 0: Text to Image with StableDiffusion")
st.subheader("Input Prompt")
st.markdown("Prompt written by the user.")

prompt_list = []
prompt = st.text_area('프롬프트를 입력하세요.')        # : 한 줄이 아닌 여러줄로 입력 받는다.   출력은 위와같이 한줄로 붙어서 된다.
prompt_list.append(prompt)
sd.sampling(prompt_list)

image = Image.open('/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/result_stable_diffusion-v1-5.png')

# Display the original image
st.image(image, caption='Result Image', width=None, use_column_width=False)

# Convert the image to a format compatible with the super resolution model
# Run super resolution with progress bar
upscaled_image = None

# Run a part of the inference]
image = image.resize((128, 128))

### 최종 때 주석 해제
upscaled_image = pipeline(image=image).images[0]
upscaled_image.save("/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/ldm_generated_image.png")

# Display the final super resolution result
st.subheader("Step 1: Super Resolution")
st.markdown("Converts the input image to a high-resolution image.")
st.image(upscaled_image, caption='Super Resolution Result', width=None, use_column_width=False)


st.markdown(" ")
st.subheader("Step 2: Segmentation")
st.markdown("Finds objects in the converted high-resolution image and segments them.")

client = Client("https://one-2-3-45-one-2-3-45.hf.space/")

segmented_img_filepath = client.predict(
'/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/ldm_generated_image.png',
api_name="/preprocess"
)
segmented_img_filepath = Image.open(segmented_img_filepath)
segmented_img_filepath.save('/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/segmented_img.png')

st.image(segmented_img_filepath, caption='Segmentation Result', width=None, use_column_width=False)

# Display the mesh file
st.markdown(" ")
st.subheader("Step 3: 3D Mesh Conversion")
st.markdown("Converts the segmented objects into a 3D mesh with the .ply extension.")
generated_mesh_filepath = client.predict(
'/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/segmented_img.png',
True,        # image preprocessing
api_name="/generate_mesh"
)
mesh = trimesh.load_mesh(generated_mesh_filepath)
target_filepath = "/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/output_mesh.ply"
shutil.copy(generated_mesh_filepath, target_filepath)

# Visualize the mesh using pydeck
st.markdown("### Congratulations!")
st.image('/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image/complete.png')

