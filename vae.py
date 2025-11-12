import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# -----------------------------
# 0) 하이퍼파라미터(필요시 수정)
# -----------------------------
EPOCHS = 5  # 학습 에폭 수
BATCH_SIZE = 128  # 미니배치 크기
LR = 2e-3  # 학습률
LATENT_DIM = 16  # 잠재공간 차원(작을수록 압축 강함)
HIDDEN1 = 512  # MLP 은닉층 크기 1
HIDDEN2 = 256  # MLP 은닉층 크기 2
BETA = 1.0  # β-VAE에서 KL 가중치(기본=1.0이면 일반 VAE)
DATA_DIR = "./data"  # 데이터셋 저장 경로
OUT_DIR = "./outputs_vae"  # 생성/복원 결과 저장 폴더

# -----------------------------
# 1) 기기/시드 설정(재현성)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# transforms.ToTensor() : [0, 255] 이미지를 0-1 범위의 float 텐서로 변환
transforms = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True, transform=transforms, download=True
)
test_dataset = datasets.MNIST(
    root=DATA_DIR, train=False, transform=transforms, download=True
)


# DataLoader : 배치 단위로 텐서를 묶어줌
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)


# 모델 정의
# 인코더 x -> mu, logvar
# 입력 x는 mnist의 B, 1, 28, 28 형태, MLP를 하려면 일단 B, 784로 펴준다.
# mu : 평균 벡터, logvar: 분산의 로그 (로그를 쓰면 수치적으로 더 안정적이다)
class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mu_head = nn.Linear(hidden2, latent_dim)
        self.logvar_head = nn.Linear(hidden2, latent_dim)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.mu_head(h)  # 평균 벡터
        logvar = self.logvar_head(h)  # 분산의 로그
        return mu, logvar


# Decoder z -> x_logits
# 출력은 시그모이드 이전의 logits를 반환, 학습은 BCEWithLogitsLoss 로
# 시각화할 때만 torch.sigmoid를 통해 0-1 확률 이미지로 바꿔 저장
class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden2=HIDDEN2, hidden1=HIDDEN1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden1)
        self.out = nn.Linear(hidden1, 28 * 28)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        logits = self.out(h)
        return logits.view(-1, 1, 28, 28)  # (B, 1, 28, 28)로 모양복원


# VAE 전체 : Encoder + Reparemeterization + decoder
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def reparameterize(mu, logvar):
        # logvar = log(sigma^2) 이므로 sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # eps - N(0, I) (mu와 std와 동일한 shape로 샘플)
        eps = torch.randn_like(std)
        # z = mu + sigma * eps
        return mu + std * eps

    def forward(self, x):
        # 인코더로 mu, logvar 추정
        mu, logvar = self.encoder(x)
        # 재파라미터라이즈로 z 샘플
        z = self.reparameterize(mu, logvar)
        # 디코더로 x_logits 생성
        x_logits = self.decoder(z)

        return x_logits, mu, logvar, z


model = VAE().to(DEVICE)

# 손실함수 (ELBO의 음수)
# 재구성 손실 : Binary Cross Entropy with Logits
# with logtis 버전을 쓰면 내부적으로 시그모이드가 포함되어 수치적으로 안정
# reduction='sum'으로 모든 픽셀/배치 합산 후, 배치 크기로 나눠 평균을 내면 배치 크기에 관계없이 손실 스케일이 일정해져 학습 안정

bce_logits = nn.BCEWithLogitsLoss(reduction="sum")


def elbo_loss(x_logits, x, mu, logvar, beta=BETA):
    # 재구성 손실 (픽셀별 BCE 합을 배치로 평균)
    recon_sum = bce_logits(x_logits, x)  # 합
    recon = recon_sum / x.size(0)  # 배치 평균

    # KL(q||p): q = N(mu, signa^2), p = N(0, I)
    # KL = -0.5 * sum(1, + logvar - mu^2 - exp(logvar))
    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_sum / x.size(0)  # 배치 평균

    loss = recon + beta * kl
    return loss, recon, kl


# 옵티마이저
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# 보조함수 (시각화/샘플링)
def save_reconstruction(model, x, epoch, out_dir=OUT_DIR, n=8):
    """입력 x와 복원 x_hat을 한 장의 그리드 이미지로 저장"""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        x = x[:n]
        x_logits, _, _, _ = model(x)
        x_hat = torch.sigmoid(x_logits)  # logits -> 확률(0~1)
        # 위쪽에 원본 n장, 아래쪽에 복원 n장
        grid = make_grid(torch.cat([x, x_hat], dim=0), nrow=n, padding=2)
        save_image(grid, os.path.join(out_dir, f"reconstruction_epoch_{epoch:03d}.png"))


def save_samples(model, epoch, out_dir=OUT_DIR, n=64):
    """p(z)~N(0,I)에서 무작위로 z를 뽑아 디코더로 이미지 생성"""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        # (n, LATENT_DIM) 모양의 표준정규 샘플
        z = torch.randn(n, LATENT_DIM, device=DEVICE)
        x_logits = model.decoder(z)
        x_prob = torch.sigmoid(x_logits)  # 시각화를 위해 확률로 변환
        grid = make_grid(x_prob, nrow=8, padding=2)
        save_image(grid, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"))


# 학습/평가 루프
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for step, (x, _) in enumerate(train_loader):
        # x: (B, 1, 28, 28)
        x = x.to(DEVICE)

        # 순전파
        x_logits, mu, logvar, _ = model(x)

        # 손실 계산
        loss, recon, kl = elbo_loss(x_logits, x, mu, logvar, beta=BETA)

        # 역전파/업데이트
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        # 몇 스텝마다 진행상황 출력

        if (step + 1) % 100 == 0:
            print(
                f"Epoch[{epoch}/{EPOCHS}] Step[{step+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.3f} (Recon : {recon.item():.3f}, KL : {kl.item():.3f})"
            )

    # 에폭 평균(스텝 기준 평균)
    steps = len(train_loader)
    print(
        f"==> Train Epoch {epoch} : "
        f"Loss {total_loss/steps:.3f} | Recon {total_recon/steps:.3f} | KL {total_kl/steps:.3f}"
    )


def evaluate(epoch):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            x_logits, mu, logvar, _ = model(x)
            loss, recon, kl = elbo_loss(x_logits, x, mu, logvar, beta=BETA)
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

    steps = len(test_loader)
    print(
        f"==> Valid Epoch {epoch}: "
        f"Loss {total_loss/steps:.3f} | Recon {total_recon/steps:.3f} | KL {total_kl/steps:.3f}"
    )


# 메인 실행부
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # (선택) 첫 배치로 빠른 sanity check (모양/손실 터지지 않는지)
    x0, _ = next(iter(train_loader))
    x0 = x0.to(DEVICE)
    model.train()
    x_logits0, mu0, logvar0, _ = model(x0)
    loss0, recon0, kl0 = elbo_loss(x_logits0, x0, mu0, logvar0, beta=BETA)
    loss0.backward()
    optimizer.step()
    print(
        f"[Sanity] Loss ~ {loss0.item():.3f} (Recon {recon0.item():.3f}, KL {kl0.item():.3f})"
    )

    # 학습 루프
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        evaluate(epoch)
        save_reconstruction(model, x0, epoch, out_dir=OUT_DIR, n=8)
        save_samples(model, epoch, out_dir=OUT_DIR, n=64)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "vae_last.pt"))
    print(f"Done. IMGS & weights saved to : {OUT_DIR}")
