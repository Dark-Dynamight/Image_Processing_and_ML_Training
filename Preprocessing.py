import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import img_as_float, exposure
from skimage.transform import resize
from skimage.util import random_noise

# 1. Format Conversion: DICOM → PNG (optional, if DICOM input)
# from pydicom import dcmread
# import SimpleITK as sitk

def resize_image(img, size=(227, 227)):
    return cv2.resize(img, size)

def normalize_image(img, method='minmax'):
    if method == 'zscore':
        return (img - np.mean(img)) / (np.std(img) + 1e-8)
    elif method == 'minmax':
        return img_as_float(img)
    else:
        raise ValueError("Unknown normalization method.")

def reduce_noise(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    median = cv2.medianBlur(img, 3)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    return gaussian, median, bilateral

def enhance_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    hist_eq = cv2.equalizeHist(img)
    gamma = 1.5
    gamma_corr = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return clahe, hist_eq, gamma_corr

def extract_breast_roi(img):
    _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = img[y:y+h, x:x+w]
        return resize_image(cropped)
    return img

def remove_pectoral_muscle(img):
    # Placeholder: simple masking (UNet recommended for accuracy)
    mask = np.ones_like(img) * 255
    pts = np.array([[0,0], [60,0], [0,60]], np.int32)
    cv2.fillPoly(mask, [pts], 0)
    return cv2.bitwise_and(img, mask)

def augment_image(img):
    flip = cv2.flip(img, 1)
    rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    brightness = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    crop = img[10:210, 10:210] if img.shape[0] > 210 else img
    return flip, rotate, brightness, crop

def extract_patches(img, patch_size=(64, 64)):
    h, w = img.shape
    patches = []
    for y in range(0, h - patch_size[1], patch_size[1]):
        for x in range(0, w - patch_size[0], patch_size[0]):
            patches.append(img[y:y+patch_size[1], x:x+patch_size[0]])
    return patches

def one_hot_label(label, classes=('benign', 'malignant', 'normal')):
    one_hot = [0] * len(classes)
    if label in classes:
        one_hot[classes.index(label)] = 1
    return one_hot

# --- Load image ---
image_path = r"C:\Users\payya\OneDrive\Documents\Major Project\Image_Processing_and_ML_Training\Images\mixed dataset\benign3.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# --- Apply preprocessing steps ---
resized = resize_image(image)
normalized = normalize_image(resized)
gauss, median, bilateral = reduce_noise((normalized * 255).astype(np.uint8))
clahe, hist_eq, gamma_corr = enhance_contrast(median)
roi = extract_breast_roi(clahe)
pectoral_removed = remove_pectoral_muscle(roi)
augmented = augment_image(pectoral_removed)
patches = extract_patches(pectoral_removed)
label = one_hot_label('benign')

# --- Display a few results ---
plt.figure(figsize=(12, 6))
plt.subplot(2,3,1); plt.imshow(image, cmap='gray'); plt.title("Original")
plt.subplot(2,3,2); plt.imshow(median, cmap='gray'); plt.title("Denoised")
plt.subplot(2,3,3); plt.imshow(clahe, cmap='gray'); plt.title("CLAHE")
plt.subplot(2,3,4); plt.imshow(roi, cmap='gray'); plt.title("ROI Extracted")
plt.subplot(2,3,5); plt.imshow(pectoral_removed, cmap='gray'); plt.title("Pectoral Removed")
plt.subplot(2,3,6); plt.imshow(augmented[0], cmap='gray'); plt.title("Augmented (Flipped)")
plt.tight_layout(); plt.show()
