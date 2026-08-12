#pip install opencv-python pytesseract pillow

import cv2
import pytesseract

# Load image
img = cv2.imread("example.png")

# 1. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Remove small black speckles/noise
# Opening removes small dark objects while preserving larger text
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
clean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# 3. Adaptive thresholding
# Useful when the background isn't perfectly uniform
binary = cv2.adaptiveThreshold(
    clean,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    10
)

# 4. Optional: remove tiny connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    255 - binary,
    connectivity=8
)

min_area = 10
filtered = binary.copy()

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area < min_area:
        filtered[labels == i] = 255

# 5. OCR
text = pytesseract.image_to_string(
    filtered,
    config="--psm 6"
)

print(text)

# Save the cleaned image so you can inspect it
cv2.imshow("Filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("cleaned.png", filtered)