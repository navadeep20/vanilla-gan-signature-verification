import torch
import os
import cv2
from generator_vanilla_gan import Generator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "generator.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "samples")

# Parameters
latent_dim = 100
num_images = 20

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Generator
G = Generator(latent_dim)
G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
G.eval()

# Generate images
for i in range(num_images):
    z = torch.randn(1, latent_dim)
    fake_img = G(z).detach().numpy()[0][0]

    # Convert from [-1,1] to [0,255]
    fake_img = (fake_img + 1) * 127.5
    fake_img = fake_img.astype("uint8")

    save_path = os.path.join(OUTPUT_DIR, f"synthetic_signature_{i+1}.png")
    cv2.imwrite(save_path, fake_img)

print("✅ Synthetic signatures generated successfully")
