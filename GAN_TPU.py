import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# 모델 정의 (생성자와 판별자)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# GAN 모델
class GAN(pl.LightningModule):
    def __init__(self, channels, width, height, latent_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        
        # 생성자와 판별자 초기화
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=(self.hparams.channels, self.hparams.width, self.hparams.height))
        self.discriminator = Discriminator(img_shape=(self.hparams.channels, self.hparams.width, self.hparams.height))
        
        # 손실함수 초기화
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_step(self, real_imgs, valid):
        # 샘플 노이즈
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)
        
        # 가짜 이미지 생성
        fake_imgs = self(z)
        
        # 판별자로 진짜와 가짜 이미지 판별
        pred_real = self.discriminator(real_imgs)
        pred_fake = self.discriminator(fake_imgs)
        
        # 손실 계산
        real_loss = self.adversarial_loss(pred_real, valid)
        fake_loss = self.adversarial_loss(pred_fake, ~valid)
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss
    
    def generator_step(self, imgs):
        # 샘플 노이즈
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        
        # 가짜 이미지 생성
        fake_imgs = self(z)
        
        # 판별자로 가짜 이미지 판별
        pred_fake = self.discriminator(fake_imgs)
        
        # 손실 계산
        g_loss = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        
        return g_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        
        valid = torch.ones(real_imgs.size(0), 1)
        valid = valid.type_as(real_imgs)
        
        # 판별자 학습
        d_loss = self.adversarial_step(real_imgs, valid)
        
        # 생성자 학습
        g_loss = self.generator_step(real_imgs)
        
        tqdm_dict = {'d_loss': d_loss, 'g_loss': g_loss}
        output = {
            'loss': d_loss if optimizer_idx == 0 else g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        
        return output
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_d, opt_g], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=64)

# 하이퍼파라미터 설정
latent_dim = 100
lr = 0.0002
epochs = 200

# GAN 모델 초기화
model = GAN(
    channels=1, 
    width=28,
    height=28,
    latent_dim=latent_dim,
    lr=lr
)

# TPU v4 사용 설정
trainer = pl.Trainer(
    accelerator="tpu", 
    devices=4,
    max_epochs=epochs
)

# 학습 시작
trainer.fit(model)