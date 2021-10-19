import cv2
import numpy as np

# cv.NORMAL_CLONE  cv.MIXED_CLONE  cv.MONOCHROME_TRANSFER

src = cv2.imread('./src.jpg', 1)
dst = cv2.imread('./dst.png', 1)
# dst = src
mask = cv2.imread('./mask.png', cv2.IMREAD_GRAYSCALE)
mask = (1 - ((mask) > 128).astype(np.uint8)) * 255
# mask = (mask+1)*255
# center = (880, 604)
center = (1920/2, 1080/2)
res = cv2.seamlessClone(src, dst, mask, center, flags=cv2.MONOCHROME_TRANSFER)
# res = cv2.seamlessClone(src, dst, mask, center, flags=cv2.NORMAL_CLONE)
cv2.imwrite('res.png', res)
