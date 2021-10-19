# /Disk1/XR/XR Color Consistency/第一帧/虚实融合.png
import cv2
import numpy as np

target = '/Disk1/XR/XR Color Consistency/第一帧/虚拟场景.png'
src = '/Disk1/XR/XR Color Consistency/第一帧/虚实融合.png'
mask = '/Disk1/XR/XR Color Consistency/第一帧/mask.png'
# '/Disk1/Projects/Blend/GP-GAN/images/test_images/底板.png'

target = '/Disk1/XR/ZRS/dst.png'
src = '/Disk1/XR/ZRS/src.jpg'
mask = '/Disk1/XR/ZRS/mask.png'


def AlphaBlend(src, target, mask):
    # 输入mask: 背景的数据为1 处理后为前景为1
    src = cv2.imread(src, 1)
    dst = cv2.imread(target, 1)
    bg_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    bg_mask=(bg_mask > 128).astype(np.float)
    fg_mask = (1 - bg_mask)  #前景为1
    kernel = np.ones((5, 5), np.uint8)
    bg_mask=cv2.dilate(bg_mask,kernel,iterations=20)  # zrs
    # bg_mask = cv2.dilate(bg_mask, kernel, iterations=30)  # 虚实
    cv2.imwrite("dilated_mask.png", bg_mask * 255)
    # blur_bg_mask=cv2.blur(bg_mask*255,ksize=(60,60))/255
    blur_bg_mask = cv2.GaussianBlur(bg_mask * 255, ksize=(101, 101),sigmaX=25) / 255
    cv2.imwrite("blur_bg_mask.png",blur_bg_mask*255)
    # blur_bg_mask=bg_mask
    res =dst*np.expand_dims(blur_bg_mask,2).repeat(3,axis=2)+ src*np.expand_dims((1-blur_bg_mask),2).repeat(3,axis=2)
    # 硬 Alpha值
    # factor=0.1                        # 表示虚拟前景需要保持的占比
    # half_mask=fg_mask*fator
    # # center = (920, 604)
    # res =dst*np.expand_dims(fg_mask*(factor)+bg_mask,2)+ src*np.expand_dims(fg_mask*(1-factor),2)
    cv2.imwrite('Alpha_fusion.png', res)

# def AlphaBlend_ratioasdistance(src, target, mask):
#     # 输入mask: 背景的数据为1 处理后为前景为1
#     src = cv2.imread(src, 1)
#     dst = cv2.imread(target, 1)
#     bg_mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
#     bg_mask= cv2.dilate(bg_mask,(3,3),iterations=5)
#     bg_mask=(bg_mask > 128).astype(np.float)
#     fg_mask = (1 - bg_mask)  #前景为1
#     bg_mask=cv2.dilate(bg_mask,(3,3),iterations=5)
#     # blur_bg_mask=cv2.blur(bg_mask*255,ksize=(60,60))/255
#
#     # blur_bg_mask=bg_mask
#     res =dst*np.expand_dims(blur_bg_mask,2).repeat(3,axis=2)+ src*np.expand_dims((1-blur_bg_mask),2).repeat(3,axis=2)
#     # 硬 Alpha值
#     # factor=0.1                        # 表示虚拟前景需要保持的占比
#     # half_mask=fg_mask*fator
#     # # center = (920, 604)
#     # res =dst*np.expand_dims(fg_mask*(factor)+bg_mask,2)+ src*np.expand_dims(fg_mask*(1-factor),2)
#     cv2.imwrite('Alpha_fusion.png', res)

AlphaBlend(src, target, mask)
