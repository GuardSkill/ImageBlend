# 针对手机拍摄图像
--src_image
/Disk1/XR/phone_texture/phone_camera.jpg
--dst_image
/Disk1/XR/phone_texture/phone_camera.jpg
--mask_image
/Disk1/XR/phone_texture/mask_label.png
--blended_image
GP_results/res_phone_GP_w2e_5.png
--color_weight
0.00002
--inpaint_bg
1

# 针对周仁爽提供的虚实图像
--src_image /Disk1/XR/ZRS/src.jpg
--dst_image /Disk1/XR/ZRS/dst.png
--mask_image /Disk1/XR/ZRS/mask.png
--blended_image GP_results/res_ZRS_GP_w8e_4.png
--color_weight 0.0008
--invert_mask 1

# 视频融合
--src_image
/Disk1/XR/xr_car
--dst_image
/Disk1/XR/xr_car
--mask_image
/Disk1/XR/car/label_mask.png
--result_folder
images/car_result_video/
--color_weight
0.00002
--inpaint_bg
1