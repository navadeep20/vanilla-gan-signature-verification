import cv2
import os

# Get absolute path of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "raw_signatures")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "signatures", "train")

IMG_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading images from:", INPUT_DIR)
print("Saving images to:", OUTPUT_DIR)

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipped: {img_name}")
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 127.5 - 1.0

    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, ((img + 1) * 127.5).astype("uint8"))

    print(f"Processed: {img_name}")

print("✅ Preprocessing finished")
