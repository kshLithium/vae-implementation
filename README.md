# MNIST용 Variational Autoencoder (VAE)

이 저장소는 PyTorch로 구현한 MNIST 손글씨용 베타-VAE 예제입니다. 단순한 MLP 인코더/디코더 구조, 재현성 설정, 출력 이미지 저장까지 학습에 필요한 최소한의 코드만 담았습니다.

## 주요 특징
- **간결한 구조:** 28×28 이미지를 입력받는 2층 MLP 인코더/디코더.
- **재파라미터라이제이션:** 평균(mu)과 로그분산(logvar)에서 잠재벡터를 샘플링.
- **β-VAE 옵션:** `BETA` 값으로 KL 가중치를 조정해 분리도(disentanglement) 실험 가능.
- **이미지 내보내기:** 에폭마다 원본/복원 그리드와 무작위 샘플 그리드를 저장.
- **재현성 도움:** 고정 시드, CPU/CUDA 자동 선택, DataLoader 결정적 옵션 사용.

## 프로젝트 구조
- `vae.py`: 데이터 전처리, VAE 모델, 손실 함수, 학습/평가 루프 정의.
- `outputs_vae/`: 학습 중 저장되는 복원/샘플 이미지와 마지막 체크포인트(`vae_last.pt`).

## 요구 사항
- Python 3.9+
- PyTorch, TorchVision (CPU 또는 CUDA 빌드)
- 인터넷(MNIST 다운로드는 최초 실행 시 자동 처리)

설치 예시:
```bash
pip install torch torchvision
```
> CUDA 버전이 필요하면 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)의 wheel 명령을 사용하세요.

## 실행 방법
학습 스크립트를 바로 실행합니다.
```bash
python vae.py
```
기본 설정으로 `./data`에 MNIST를 받으며 5에폭 학습 후 출력은 `./outputs_vae/`에 생성됩니다. 두 경로가 없다면 자동으로 만들어지니 따로 준비할 필요가 없습니다.
- `reconstruction_epoch_XXX.png`: 한 배치의 원본(윗줄)과 복원 이미지(아랫줄) 그리드.
- `samples_epoch_XXX.png`: 잠재공간에서 무작위로 샘플한 이미지 그리드.
- `vae_last.pt`: 마지막 모델 가중치.

## 하이퍼파라미터 요약 (`vae.py` 상단)
- `EPOCHS`: 학습 에폭 수 (기본 5)
- `BATCH_SIZE`: 배치 크기 (기본 128)
- `LR`: Adam 학습률 (기본 2e-3)
- `LATENT_DIM`: 잠재공간 차원 (기본 16)
- `HIDDEN1` / `HIDDEN2`: MLP 은닉층 크기 (기본 512 / 256)
- `BETA`: KL 가중치(β-VAE) (기본 1.0)
- `DATA_DIR` / `OUT_DIR`: 데이터/출력 경로

## 출력 확인 및 모니터링
콘솔에는 스텝별/에폭별 재구성 손실과 KL 손실 평균이 표시됩니다. `outputs_vae/`의 이미지 그리드를 통해 복원 품질과 생성 다양성을 시각적으로 확인하세요.

## 확장 아이디어
- 인코더/디코더를 합성곱 네트워크로 바꿔 이미지 품질을 높이기.
- `LATENT_DIM`이나 은닉 크기를 늘려 더 풍부한 표현 학습하기.
- 다른 데이터셋이나 변환을 적용해 도메인 확장 실험하기.
