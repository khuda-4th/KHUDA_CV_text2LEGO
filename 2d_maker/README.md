## 2d maker

3d mesh(.off)에서 카메라 방향 조절을 통한 2d rendering

### 1. ModelNet40의 데이터셋 사용
[다운로드 링크](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset)
이 중 14개의 클래스를 선별적으로 골라 사용하였습니다.(선별한 클래스는 .py 참고)

### 2. 카메라 방향 조절 설명

```python
z_model_center = (model_extents[0] + model_extents[1]) / ? + max_bound[2] / ?

pose = np.array([
    [1.0, 0.0, 0.0, density_center[0]],
    [0.0, 1.0, 0.0, density_center[1] - (model_extents[0] + model_extents[1]) / ?],
    [0.0, 0.0, 1.0, z_model_center],
    [0.0, 0.0, 0.0, 1.0]
])

yaw, pitch, roll = np.radians(0), np.radians(0), np.radians(?)
```

#### camera pose 설명
1행은 camera의 x좌표<br>
2행은 camera의 y좌표<br>
3행은 camera의 z좌표<br>
4행은 필수적으로 [0.0, 0.0, 0.0, 1.0]로 설정되어야 합니다.<br>
<br>
yaw, pitch, roll은 순서대로 x축, y축, z축의 각도를 조절합니다.<br>
여기서는 z축의 각도만 조절하면 됩니다.<br>
<br>
? 부분을 각 object의 특성에 맞추어 조절하면 됩니다.<br>

