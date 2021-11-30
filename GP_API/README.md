# GP图像融合接口介绍

## API调用说明

### 1.初始化类/模型

```python
gp_model = GPU_GP_Container(img_shape=(1, 3, *mask.shape), color_weight=args.color_weight, gpu=args.gpu)
```

类参数如下：

```python
img_shape=(1,B,H,W)      目前需要将模型固化为（1,3,H,W）的分辨率
color_weight= 0.5        指定前景的颜色比重()
sigma=0.5                高斯核的sigma，目前此项废弃（无效）
gpu=0                    指定CUDA的设备,-1表示cpu
```

### 2.融合请调用类中的推导API

```python
blended_ims = gp_model.GP_GPU_Model_fusion(obj, bg, mask, args.gpu)
```

参数如下：

```python
obj                    numpy形式的前景图像（H,W,3）
bg                     numpy形式的背景图像（H,W,3）
mask    		       numpy形式的Mask图像 （H,W）,指定了前景的需要抠出的区域
gpu=0                  指定CUDA的设备
```



调用的方式可以参照`Test_API.py`，其命令参数可以参照`test_command.sh`。