import torch
import torch.nn as nn
import torch.optim as optim
import random
from data_loader_signatures import get_signature_dataloader

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset path (AUGMENTED DATA)
# -------------------------------
TRAIN_DIR = "../data/signatures_augmented/train"


# -------------------------------
# Signature Verification Model
# -------------------------------
class SignatureVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Training Logic
# -------------------------------
def train_verifier():
    # Load AUGMENTED data (real + synthetic)
    dataloader = get_signature_dataloader(
        batch_size=1,
        data_dir=TRAIN_DIR
    )

    images = [img for img, _ in dataloader]

    model = SignatureVerifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        total_loss = 0.0

        for img in images:
            img = img.to(device)

            # Genuine pair (same image twice)
            genuine_pair = torch.cat([img, img], dim=1)
            genuine_label = torch.ones((1, 1), device=device)

            # Forged pair (random mismatched image)
            forged_img = random.choice(images).to(device)
            forged_pair = torch.cat([img, forged_img], dim=1)
            forged_label = torch.zeros((1, 1), device=device)

            for pair, label in [
                (genuine_pair, genuine_label),
                (forged_pair, forged_label)
            ]:
                optimizer.zero_grad()
                output = model(pair)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "../checkpoints/signature_verifier_augmented.pth")
    print("✅ Augmented signature verifier trained and saved")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train_verifier()
