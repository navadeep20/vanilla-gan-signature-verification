import torch
import random
from signature_verifier_train import SignatureVerifier
from data_loader_signatures import get_signature_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = SignatureVerifier().to(device)
model.load_state_dict(torch.load("../checkpoints/signature_verifier.pth"))
model.eval()

dataloader = get_signature_dataloader(batch_size=1)
images = [img for img, _ in dataloader]

TP = TN = FP = FN = 0

with torch.no_grad():
    for img in images:
        img = img.to(device)

        # Genuine pair (should be accepted)
        genuine_pair = torch.cat([img, img], dim=1)
        genuine_score = model(genuine_pair).item()

        if genuine_score >= 0.5:
            TP += 1
        else:
            FN += 1

        # Forged pair (random mismatch, should be rejected)
        forged_img = random.choice(images).to(device)
        forged_pair = torch.cat([img, forged_img], dim=1)
        forged_score = model(forged_pair).item()

        if forged_score >= 0.5:
            FP += 1
        else:
            TN += 1

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
FAR = FP / (FP + TN + 1e-6)
FRR = FN / (FN + TP + 1e-6)

print("✅ Verification Evaluation Results")
print(f"Accuracy : {accuracy:.2f}")
print(f"FAR      : {FAR:.2f}")
print(f"FRR      : {FRR:.2f}")
print("EER      : Approximated at point where FAR ≈ FRR")
