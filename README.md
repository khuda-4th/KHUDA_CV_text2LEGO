# Text2LEGO : Building LEGO 3D blueprint with only text Using Generative AI model

## 연구 동기

저희는 **원하는 모든 것을 LEGO로 만들어보자!** 라는 아이디어로 이 프로젝트를 시작했습니다. 하지만 사용자가 만들고 싶은 물체의 사진을 인터넷 상에서 구할 수 없을 수 있기 때문에 Text to image 모델을 결합하여 텍스트만으로 만들고 싶은 물체의 사진으로 레고 도면을 제작해주는 서비스를 고안하였습니다.

## 연구 과정

### Architecture
![architecture](https://github.com/khuda-4th/text2LEGO/assets/108567536/b0f444db-b19a-48d7-9c91-72cbaca85817)
먼저 프롬프트를 통해 텍스트를 입력받고, 이를 text2image Stable diffusion 모델을 통해 2D image로 만들고, Super resolution을 통해 고해상도의 이미지로 변환됩니다. 이렇게 만들어진 고해상도의 이미지는 One-2-3-45 모델을 통해 segmentation되고 3D mesh로 변환됩니다. 이를 복셀화하고 Lego algorithm을 거치면 레고 도면이 생성됩니다. 

### Image2LEGO
![image2lego](https://github.com/KHAI-2023/Make_Anything_with_LEGO/assets/64340624/f9ab4104-4da6-4866-a0b9-6bb31cfde274)
저희가 기존에 채택했던 구조는 Image2LEGO 모델로 아키텍쳐는 그림과 같습니다. 이는 크게 3d autoencoder와 2d image encoder로 나뉩니다. 3d object를 3d encoder에 넣고, 2d image를 2d encoder에 넣어 나온 각각의 latent vector들 간의 MSE loss를 통해 3d encoder를 학습시킵니다. 이후 input 값인 3d object와 decoder를 통해 나온 3d object output 사이의 BCE Loss를 구하고 가중치를 갱신하는 과정을 통해 네트워크를 학습시킵니다. train 과정이 끝난 이후에는 학습 시킨 2d encoder와 3d decoder를 결합합니다. 결과적으로 2d image를 넣었을 때 3d object가 나오는 test 모델을 완성할 수 있습니다.

그러나 위 모델을 시도한 결과 알아낸 한계점은 크게 두가지입니다.

첫번째로 범용성의 문제입니다. 이 모델은 학습에 사용된 데이터의 카테고리의 물체들만 제대로 된 3d object로 생성할 수 있습니다. 또한 반드시 배경이 없는 사진을 넣어야 작동 가능하기 때문에 일반 유저가 쉽게 사용하기 어렵다는 문제점이 있습니다. 

두번째로 일반 유저에게 서비스로 제공하기에 inference하는 시간이 오래 걸린다는 단점이 있습니다. 

이를 보완하기 위하여 저희는 새로운 모델을 채택하였습니다.

### One-2-3-45
<img width="807" alt="Screenshot 2023-11-21 at 3 52 43 PM" src="https://github.com/KHAI-2023/Make_Anything_with_LEGO/assets/64340624/7f2c1422-856d-4a8d-a872-f354f0f700eb">

저희가 새롭게 채택한 모델은 One-2-3-45 모델입니다.

이는 기존 3D reconstruction 모델에서의 4가지 한계를 보완하여 만들어진 모델입니다. 그 4가지 문제는 2d image에서 3d object로 만들기까지 많은 iteration이 필요하기 때문에 발생하는 time-consuming 문제, image의 해상도가 커질수록 발생하는 memory intensive 문제, 3d reconstruction 과정에서 2d image의 가려진 부분도 추론해서 만들어 기대한 바와는 다른 3d 출력 결과가 나오는 3d inconsistent문제, 마지막으로 high-quality의 mesh를 만들지 못하는 poor geometry 문제입니다.

위 모델은 앞선 4가지의 문제를 어느정도 해결해 줌과 더불어, inference time을 45초로 줄였다는 장점이 있습니다. 먼저, 입력된 multi-view synthesis 모듈에서 zero123이라는 2d difffusion 모델을 활용해 다양한 시점에서의 2d 이미지를 생성해냅니다. 이후 near by한 4개의 view를 활용하여 pose estimation 모듈을 진행합니다. 마지막으로 multi view posed image들을 통해 360도의 mesh reconstruction모듈을 수행합니다.

one-2-3-45 모델은 앞선 세 모듈의 결합으로 불필요한 optimization 과정을 간소화했고, 한 번의 feed-forward만으로도 3d mesh를 적은 시간 내에 구현할 수 있었습니다.

### Text to image(Stable Diffusion)

저희는 Text to image를 Stable Diffusion 모델을 통해 구현했습니다. Stable diffusion 은 VAE 를 사용하여 이미지를 저차원으로 사영하고 저차원에서의 diffusion process 를 수행합니다. 저차원에서 text 와 이미지 정보를 Cross Attention 방식으로 연결시킨 후 샘플링된 저차원 벡터를 VAE 디코더에 입력하여 이미지를 생성합니다. 이렇게 저차원에서 diffusion process 를 거치면 computational cost 를 줄일 수 있습니다.

### Super Resolution
<img width="807" alt="Screenshot 2023-11-21 at 3 53 51 PM" src="https://github.com/KHAI-2023/Make_Anything_with_LEGO/assets/64340624/774d9205-6432-4c4c-b69b-eb49f0d82dda">

또한 저희는 기존의 one-2-3-45 모델에 Super resolution 기법까지 결합해 보았습니다. 사용자들이 인터넷을 통해 흔히 구할 수 있는 사진은 저해상도이거나 일부 픽셀이 왜곡되어있는 경우가 많습니다. 그렇기에 2D to 3D reconstruction을 진행할 때 픽셀 왜곡에 의해 의도치 않았던 Object가 생성되는 것을 막고자 Image Segmentation을 진행하기 이전에 Super Resolution 과정을 거쳤습니다.

<img width="1001" alt="diffusionmodel" src="https://github.com/KHAI-2023/Make_Anything_with_LEGO/assets/127406760/c1831c3b-1e19-4e3b-b6d8-fdad01a24062">

저희가 super resolution에서 사용한 모델 stable diffusion 모델입니다. 해당 모델은 기존의 diffusion model들과는 달리,  autoencoder구조를 적용하여 pixel 공간과 perceptual하게 동일한 latent space를 학습합니다. 이는 space를 압축할 필요가 없어 계산 복잡성을 줄일 수 있고, 효율적인 학습을 통해 dall-e나 vqgan에 비해 개선된 성능을 보이기도 했습니다.

### SR + One-2-3-45
<img width="807" alt="Screenshot 2023-11-21 at 3 57 16 PM" src="https://github.com/KHAI-2023/Make_Anything_with_LEGO/assets/64340624/d32e7492-a666-4eac-8970-fe18c4045d68">

앞서 설명한 모든 내용을 합친 최종적인 저희의 아키텍처는 다음과 같습니다. 우선 stable diffusion 모델을 활용하여 저해상도의 사진을 고해상도의 사진으로 upscaling해주고 denoising 해주었습니다. 이를 one-2-3-45 모델을 통해 Segmentation을 거쳐 이미지의 object만을 추출한 뒤, 추출된 2d image를 3d mesh 형태로 생성합니다. 이를 voxel화 한후 ColouredVoxels2LDR이라는 모델을 통하여 최종적으로 LEGO 도면을 생성합니다. 

### Colouredvoxels2LDR
저희의 주요 모델을 통해 생성된 3d mesh 이미지는 Colouredvoxels2LDR 알고리즘을 통해 레고 도면으로 완성됩니다. 위 알고리즘은 우선 각각의 복셀을 1*1의 레고 브릭으로 변환합니다. 그리고 모델의 Z좌표에서 각 레이어를 반복하여 브릭을 더 큰 브릭으로 결합할 그룹을 찾습니다. 컬러 정보의 경우에는 voxel 파일에서 추출한 후 레이어를 최적화하여 레고 브릭에 적용하게 됩니다. 

## 결과
### 시연영상


### 연구적 가치
- Text2LEGO 모델은 Super Resolution 과 One-2-3-45 모델에 연결하여 더 좋은 output을 추출할 수 있었습니다. 또한 기존 Image2LEGO 모델에 비해 단순한 3d public dataset의 카테고리에서 벗어나 더 많은 결과물을 만들 수 있습니다.
- 또한 text-to-image 모델과 결합하여 text로 원하는 LEGO 도면을 추출할 수 있습니다.

### 상업적 가치
한국콘텐츠진흥원의 연구 결과에 따르면 키덜트 시장 규모는 2014년 5000억 원 수준에서 2020년 1조 6000억 원으로 확대되었고, 이는 향후 최대 11조 원까지 성장할 것으로 전망된다고 말합니다. 저희는 이렇듯 상업적 가치가 있는 키덜트 시장에 포커스를 맞추었습니다. 

우선 자신만의 세상을 구축하기 원하는 키덜트들에게 저희의 모델을 서비스화하여 성장하는 시장에서 다수의 이용자들을 유치할 수 있을것입니다.

또한 사용자가 원하는 디자인의 이미지를 통해 LEGO 도면을 제작하여 저비용으로 레고 디자인을 할 수도 있습니다. 네버랜드 신드롬 트렌드에 따라 다시 인기를 끄는 캐릭터들의 레고 도면을 제작해 대중의 이목을 끌고 마케팅적으로도 활용할 수 있을 것입니다. 

## 참고문헌
[1] Image2Lego: Customized LEGO Set Generation from Images, https://arxiv.org/pdf/2108.08477.pdf

[2] High-Resolution Image Synthesis with Latent Diffusion Models, https://arxiv.org/pdf/2112.10752.pdf

[3] One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization, https://arxiv.org/pdf/2306.16928.pdf

[4] Colouredvoxels2LDR, https://github.com/pennyforge/ColouredVoxels2LDR


## References
* ColouredVoxel2LDR: https://github.com/pennyforge/ColouredVoxels2LDR
* One-2-3-45: https://github.com/One-2-3-45/One-2-3-45
* Diffusers: https://github.com/huggingface/diffusers




