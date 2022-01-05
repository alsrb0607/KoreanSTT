<div align="center">

## 외국인 발화 한국어 음성 인식(Korean STT)


**kospeech를 활용한 한국어 음성인식 모델 개발**
___
</div>

해당 프로젝트는 End-to-End 한국어 음성 인식 오픈소스 툴킷인
[kospeech](https://github.com/sooftware/kospeech)를 활용하여 진행했음을 밝힙니다.
</br></br>

## - 프로젝트 개요
- 외국인 발화 한국어 음성 데이터를 활용해 음성 인식 모델을 개발하고, 관련 사업화 아이디어를 제안하는 해커톤에 참여한 프로젝트의 일부입니다. </br>(대회 기간: 약 3주)
- 위의 이유로, 데이터와 학습을 완료한 모델은 업로드 하지 않았습니다.
- 총 30만 개의 학습 데이터를 활용하여 모델을 개발하고 1만 개의 테스트 데이터에 대해 CER, WER을 측정한 결과 'CER: 0.085, WER: 0.1919'의 결과를 얻었습니다.
- 모델 성능 40%, 사업화 아이디어 60%의 평가에서 전체 29개 참여 팀 중 성능 부문 1위, 최종 평가 2위로 우수상을 수상했습니다.
- 원본 오픈소스 코드에 에러/버그가 꽤 많았습니다. 해당 문제들을 하나씩 해결한 결과물이고, 그 과정에서 제가 사용한 데이터에 맞게 변형된 부분이 존재할 수 있습니다.
- 자세한 오픈소스 활용 방법과 그 과정에서 발생한 문제들을 해결한 과정은 제 [블로그](https://mingchin.tistory.com/152)의 'kospeech(한국어 STT)' 카테고리를 참고하시기 바랍니다.
- ./bin/inference_wer.py, ./bin/tools.py, ./bin/prediction.py는 kospeech에 존재하지 않으며, 필요에 의해 생성한 파일입니다.
- 꼭 필요하지는 않지만 참고가 될만한 파일은 ./etc 안에 두었습니다.
</br></br>

## - 모듈 설치
```
!pip install -r requirements_cssiri.txt
```
- Python 3.8을 사용했습니다. 
- kospeech가 제공하는 다양한 Acoustic Model 중, ds2(deepspeech2)를 사용했습니다.
- Pytorch의 경우 1.10 버전이 사용되기 때문에 상위 버전을 사용하시는 경우 별도로 Pytorch를 재설치해주어야 합니다.
- 전처리, 학습, 예측, 예측한 결과 저장에 필요한 모든 모듈을 포함시켰습니다.
</br></br>

## - 전처리(Preprocess)
```
!python ./dataset/kspon/main.py --dataset_path $dataset_path --vocab_dest $vacab_dict_destination --output_unit 'character' --preprocess_mode 'phonetic' 
```
- output_unit과 preprocess_mode는 상황에 맞게 지정해주시면 됩니다.
- ./dataset/kspon/preprocess/preprocess.py의 line 95~101을 확인해보시면, './'의 위치에 'train.txt' 파일을 필요로 합니다. 해당 파일은 '음성 파일 경로' + '\t' + '한국어 전사' 의 형식으로 작성되어야 합니다.
- train.txt를 만들 때 사용한 코드는 ./etc/traintext 생성.ipynb 에 올려두었습니다.
</br></br>

## - 학습(Train)
```
!python ./bin/main.py model=ds2 train=ds2_train train.dataset_path=$dataset_path
```
- 학습과 관련된 configs(epoch, batch_size, spec_augment, 음성 파일 확장자 등)의 수정은 ./configs/audio/fbank.yaml 혹은 ./configs/train/ds2_train.yaml 에서 하실 수 있습니다. 
</br>보다 자세한 설명은 [여기](https://mingchin.tistory.com/222)를 참고해주세요.
</br></br>

## - 예측(Inference)

모든 command에 대한 deivice 옵션은 상황에 맞게 지정해주세요.

1. 음성 파일 1개에 대한 예측
* Command

```
!python ./bin/inference.py --model_path $model_path --audio_path $audio_path --device "cpu"
```
* Output

```
음성 인식 결과
```
2. 음성 파일 1개에 대한 예측과 Cer, Wer 계산 결과 저장</br>
(결과는 dst_path에 저장되며, 정답 label인 transcripts.txt파일을 transcript_path에 지정해주어야 합니다. 그 형식은 전처리에 필요한 train.txt 파일 혹은 학습에 사용되는 transcripts.txt와 동일해야 합니다.)
* Command
```
python ./bin/inference_wer.py --model_path $model_path --audio_path $audio_path --transcript_path $transcript_path --dst_path $result_destination --device "cpu"
```
* Output

```
음성 인식 결과
```
3. 음성 파일 여러 개(폴더)에 대한 예측과 그 결과 저장(.txt, .xlsx)
* Command
```
python ./bin/prediction.py --model_path $model_path --audio_path $audio_path --submission 'True' --device "cpu"
```
'submission = True'로 지정하면 예측 결과를 .xlsx 파일로 저장할 수 있습니다. 다만 2개의 컬럼을 갖는 제출용 excel 파일을 필요로 합니다.
* Output

./outputs 폴더에 .txt와 .xlsx 파일 생성
</br></br>

## - References
- Wer, Cer 관련: 
https://holianh.github.io/portfolio/Cach-tinh-WER/
- kospeech:
https://github.com/sooftware/kospeech
