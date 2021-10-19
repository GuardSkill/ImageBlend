# /Disk1/XR/XR Color Consistency/第一帧/虚实融合.png
import cv2
import numpy as np

case=4
if case==1:
    target = '/Disk1/XR/XR Color Consistency/第一帧/虚拟场景.png'
    src = '/Disk1/XR/XR Color Consistency/第一帧/虚实融合.png'
    mask = '/Disk1/XR/XR Color Consistency/第一帧/mask.png'
elif case == 2:
    target = '/Disk1/Projects/Blend/GP-GAN/images/test_images/res_colorw2.png'
    src = target
    mask = '/Disk1/Projects/Blend/GP-GAN/images/test_images/底板.png'
elif case == 3:
    target='/Disk1/XR/phone_texture/phone_camera.jpg'
    src = '/Disk1/XR/phone_texture/phone_camera.jpg'
    mask = '/Disk1/XR/phone_texture/mask_label.png'
else:
    target = '/Disk1/XR/ZRS/dst.png'
    src = '/Disk1/XR/ZRS/src.jpg'
    mask = '/Disk1/XR/ZRS/mask.png'


def PoissonBlend(src, target, mask):
    # 输入的mask背景为1 前景为0
    src = cv2.imread(src, 1)
    dst = cv2.imread(target, 1)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = (1 - (mask > 128).astype(np.uint8)) * 255
    # center = (920, 604)   # 虚实
    center = (880, 604)    # ZRS
    # center = (int(1920 / 2), int(1080 / 2))
    # center = (721, 893)
    # 输入的mask背景为0 前景为1
    # res = cv2.seamlessClone(src, dst, mask, center, flags=cv2.MONOCHROME_TRANSFER)
    res = cv2.seamlessClone(src, dst, mask, center, flags=cv2.NORMAL_CLONE)
    cv2.imwrite('bossionBlend_2fusion_phone.png', res)

#  compute center
# top, bottom, left, right = 20, 20, 20, 20
# border_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
#
# contours, _ = cv2.findContours(border_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# c = max(contours, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(c)
# center = (int(x + w * 0.5), int(y + h * 0.5))



# import Algorithmia
#
# input = ''
# client = Algorithmia.client('YOUR_API_KEY')
# algo = client.algo('wuhuikai/blending/0.2.0')
# algo.set_options(timeout=300) # optional
# print(algo.pipe(input).result)
#
#
PoissonBlend(src, target, mask)
