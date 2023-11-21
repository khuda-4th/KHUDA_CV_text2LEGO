import os
import trimesh
import pyrender
import numpy as np
from PIL import Image
import transforms3d as t3d

# ModelNet40 데이터 폴더 경로
modelnet40_folder = 'ModelNet40/piano'

def calculate_density_center(trimesh_mesh):
    vertices = trimesh_mesh.vertices  # 3D 모델의 모든 정점 위치

    if len(vertices) == 0:
        return [0.0, 0.0, 0.0]

    density_center = vertices.sum(axis=0) / len(vertices)
    return density_center

# Train 및 Test 폴더 생성
os.makedirs(os.path.join(modelnet40_folder, '2d_train'), exist_ok=True)
os.makedirs(os.path.join(modelnet40_folder, '2d_test'), exist_ok=True)

category_path = os.path.join(modelnet40_folder)

for split in ['train', 'test']:
    data_folder = os.path.join(modelnet40_folder, split)
    output_split = os.path.join(category_path, f"2d_{split}")

    # .off 파일을 렌더링하여 2D 이미지로 저장
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.off'):
            off_file_path = os.path.join(data_folder, file_name)
            trimesh_mesh = trimesh.load(off_file_path)
            mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)

            # 3D 모델의 크기와 중심을 계산
            min_bound, max_bound = mesh.bounds
            model_extents = trimesh_mesh.extents
            density_center = calculate_density_center(trimesh_mesh)
                
            # 원하는 시점 및 방향을 나타내는 카메라 포즈 설정
            z_model_center = (model_extents[0] + model_extents[1]) / 3.0 + max_bound[2] / 2.8
            pose = np.array([
                [1.0, 0.0, 0.0, density_center[0]],  # X 좌표 조정
                [0.0, 1.0, 0.0, density_center[1] - (model_extents[0] + model_extents[1] + model_extents[2]) / 2.8],  # Y 좌표 조정
                [0.0, 0.0, 1.0, z_model_center],  # Z 좌표 조정
                [0.0, 0.0, 0.0, 1.0]
            ])

            # Yaw, pitch, roll angles in radians
            yaw, pitch, roll = np.radians(0), np.radians(0), np.radians(60)

            rotation_matrix = t3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')[:3, :3]

            pose[:3, :3] = 1.25 * rotation_matrix  # Apply rotation

            # 방향성 조명 설정
            directional_light = pyrender.light.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

            camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)

            scene = pyrender.Scene()
            scene.add(mesh)
            scene.add(camera, pose=pose)
            scene.add(directional_light, pose=pose)

            r = pyrender.OffscreenRenderer(128, 128)
            color, _ = r.render(scene)
            r.delete()

            # 이미지를 해당 클래스와 분할에 따라 저장
            jpg_filename = os.path.splitext(file_name)[0] + ".jpg"
            output_path = os.path.join(output_split, jpg_filename)
            image = Image.fromarray(np.uint8(color))
            image.save(output_path)
            print(f"Rendering and saving complete: {output_path}")

print("2D 렌더링이 완료되었습니다.")
