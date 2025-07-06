# #DICOM Image import
# import pydicom
# import matplotlib.pyplot as plt
# import cv2

# # Load DICOM image
# ds = pydicom.dcmread("C:/Users/payya/OneDrive/Documents/Major Project/Images/1-1.dcm")
# image = ds.pixel_array

# plt.imshow(image, cmap='gray')
# plt.title("Original DICOM Image")
# plt.show()


#png Image import
import cv2
import matplotlib.pyplot as plt

# Load PNG in grayscale
image = cv2.imread("C:/Users/payya/OneDrive/Documents/Major Project/Images/Dicom-Systems_Mammography_Breast_Imaging_Workflows_3D_Tomosynthesis.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.title("Original PNG Image")
plt.axis('off')
plt.show()
# Wait until any key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

