import trimesh
import shutil
import os

# 대상 폴더 경로
target_folder = 'folder'

# 메쉬 로드
mesh = trimesh.load_mesh(generated_mesh_filepath)

# 대상 폴더에 저장할 파일 경로
target_filepath = os.path.join(target_folder, os.path.basename(generated_mesh_filepath))

# 대상 폴더에 복사
shutil.copy(generated_mesh_filepath, target_filepath)

print(f'Mesh 파일이 {target_filepath}로 복사되었습니다.')