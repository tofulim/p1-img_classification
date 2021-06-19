# pstage_02_image_classification
- 기간 : 2021.03.29~2021.04.09
- 대회 내용 : 이미지에서 연령대/성별/마스크 착용 상태를 파악하는 대회(acc : 81.6984% / f1 score : 0.7689 최종 5등) 
- 수행 요약 : 가벼운 모델의 분류기만으로 실험 및 세팅 후 freeze를 풀고 여러 모델들에 적용, pytorch, timm을 통해 pre-trained Resnet18/50, EfficientNet_b3, Resnext, Nfnet이용

## Source
### notebook환경으로 진행
- dataset : load dataset, labeling, augmentation experiment
- main : get model, train, inference, ensemble
- multi_classification : moduel for multi_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Importance Technic
- multi-head classifier (split 18 classes target to mask/age/gender)
- hard voting 

### Ensemble figure
![앙상블최종](https://user-images.githubusercontent.com/52443401/120945749-29200f00-c775-11eb-9e22-bf54e2c8a4b4.JPG)

모델간의 클래스 분포 차이가 클 때 앙상블의 재료로 사용했습니다.
