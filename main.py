import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
RESULT_DIR = "results"

def load_images():
    images = []
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(DATASET_DIR, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append((filename, img))
    return images

def dummy_segmentation(img):
    # Simulate segmentation using grayscale + thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask

def save_results(images):
    os.makedirs(RESULT_DIR, exist_ok=True)
    for filename, img in images:
        mask = dummy_segmentation(img)
        result_path = os.path.join(RESULT_DIR, f"segmented_{filename}")
        cv2.imwrite(result_path, mask)
        print(f"Saved: {result_path}")

if __name__ == "__main__":
    print("Loading images...")
    images = load_images()
    print(f"Found {len(images)} images.")
    print("Applying segmentation...")
    save_results(images)
    print("Done!")
