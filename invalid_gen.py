import cv2
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
INPUT_IMAGE = r"C:\Users\paran\OneDrive\Desktop\Final Year Project\Phase2\annotation_files_and_images\JPEGImages\img000032.jpg"   # Change this to your image
OUTPUT_DIR = r"C:\Users\paran\OneDrive\Desktop\hackathon_karur\invalid_test_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Overexposure
# -----------------------------
def overexpose(img, factor=2.5):
    # Increase brightness drastically
    over = np.clip(img.astype(np.float32) * factor, 0, 255)
    return over.astype(np.uint8)


# -----------------------------
# Underexposure
# -----------------------------
def underexpose(img, factor=0.25):
    # Darken heavily
    under = np.clip(img.astype(np.float32) * factor, 0, 255)
    return under.astype(np.uint8)


# -----------------------------
# Very High Blur
# -----------------------------
def high_blur(img, k=35):
    # Gaussian blur with large kernel
    return cv2.GaussianBlur(img, (k, k), 0)


# -----------------------------
# Salt & Pepper Noise (optional)
# -----------------------------
def salt_pepper(img, amount=0.02):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)

    # Salt (white dots)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper (black dots)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


# -----------------------------
# RUN AUGMENTATIONS
# -----------------------------
img = cv2.imread(INPUT_IMAGE)
if img is None:
    print("Error: input image not found!")
    exit()

# BGR → RGB optional (your choice)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

aug1 = overexpose(img)
aug2 = underexpose(img)
aug3 = high_blur(img)
aug4 = salt_pepper(img)

# Save outputs
cv2.imwrite(os.path.join(OUTPUT_DIR, "overexposed.jpg"), cv2.cvtColor(aug1, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "underexposed.jpg"), cv2.cvtColor(aug2, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "high_blur.jpg"), cv2.cvtColor(aug3, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUTPUT_DIR, "salt_pepper.jpg"), cv2.cvtColor(aug4, cv2.COLOR_RGB2BGR))

print("Augmentation complete! Saved to:", OUTPUT_DIR)
