# # Using OpenCV
# import cv2

# # Resize to 512x512 (adjust based on model input size)
# resized_image = cv2.resize(image, (512, 512))

# print(resized_image.shape)  # Should be (512, 512)



# # Using scikit-image
# from skimage.transform import resize

# # Resize and convert pixel values to float [0, 1]
# resized_image = resize(image, (512, 512), anti_aliasing=True)

# print(resized_image.shape)  # (512, 512)


import pydicom
import cv2

ds = pydicom.dcmread("C:/Users/payya/OneDrive/Documents/Major Project/Image_Processing_and_ML_Training/Images/1-1.dcm")
image = ds.pixel_array

# Normalize if required before resize
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Resize
resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
