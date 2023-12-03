
import torch
from PIL import Image
import os
from diffusers import StableDiffusionPipeline

def get_prompt(file_name="./prompt.txt"):
    prompts = []

    with open(file_name, "r") as file:
        lines = file.readlines()
        for line in lines:
            prompts.append(line.strip())
    return prompts


def sampling(prompt, batch_size=8):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  
    for i in range(len(prompt)):
        images = pipe(prompt=prompt, num_inference_steps=1).images

        if not os.path.exists("results"):
            os.makedirs("results")

        file_name = f"result_stable_diffusion-v1-5.png"
        file_path = '/Users/hyeokseung/Desktop/mawl/Make_Anything_with_LEGO/image'

        images[0].save(os.path.join(file_path, file_name))
        print(f"Image saved: {file_path}")
