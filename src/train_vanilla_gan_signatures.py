import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generator_vanilla_gan import Generator
from discriminator_vanilla_gan import Discriminator
import os

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
latent_dim = 100
batch_size = 4   # small because we have few images
epochs = 50
lr = 2e-4

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset & DataLoader
dataset = datasets.ImageFolder(
    root="../data/signatures",
    transform=transform
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# Models
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

# Loss & Optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for imgs, _ in loader:
        imgs = imgs.to(device)

        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = G(z)

        real_loss = criterion(D(imgs), valid)
        fake_loss = criterion(D(fake_imgs.detach()), fake)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        g_loss = criterion(D(fake_imgs), valid)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Save model
os.makedirs("../checkpoints", exist_ok=True)
torch.save(G.state_dict(), "../checkpoints/generator.pth")

print("✅ Training finished and model saved")
