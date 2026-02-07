import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "samples")

pixel_counts = []

for img_name in os.listdir(SAMPLES_DIR):
    img_path = os.path.join(SAMPLES_DIR, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # Count dark pixels (ink density)
    pixel_count = np.sum(img < 200)
    pixel_counts.append(pixel_count)

print("Number of generated samples:", len(pixel_counts))
print("Minimum stroke pixels:", min(pixel_counts))
print("Maximum stroke pixels:", max(pixel_counts))
print("Average stroke pixels:", int(np.mean(pixel_counts)))
