from gradio_client import Client
client = Client("https://one-2-3-45-one-2-3-45.hf.space/")
# example input image
input_img_path = "https://huggingface.co/spaces/One-2-3-45/One-2-3-45/resolve/main/demo_examples/01_wild_hydrant.png"

generated_mesh_filepath = client.predict(
   input_img_path,   
   True,      # image preprocessing
   api_name="/generate_mesh"
)

elevation_angle_deg = client.predict(
   input_img_path,
   True,      # image preprocessing
   api_name="/estimate_elevation"
)

segmented_img_filepath = client.predict(
   input_img_path,   
   api_name="/preprocess"
)

# seg 저장
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 세그멘테이션된 이미지를 로드합니다.
segmented_img = mpimg.imread(segmented_img_filepath)

# 세그멘테이션된 이미지를 표시합니다.
plt.imshow(segmented_img)
plt.axis('off')

# 저장할 이미지 파일 경로를 지정합니다.
output_image_filepath = 'output_segmented_image6.png'

# 이미지를 지정된 경로에 저장합니다.
plt.savefig(output_image_filepath, bbox_inches='tight')

print(f"Segmented image saved to: {output_image_filepath}")