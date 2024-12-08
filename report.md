# FFTDORMER-REPLICATION
本仓库复现的论文为 https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf

对应的代码为 https://github.com/kkkls/FFTformer

所有的复现实验都是在一张 NVIDIA A6000 GPU 上完成的

原论文只在 GoPro 等常规数据集上进行了训练和测试，我在复现这些数据集的同时还在 UHD-Blur 这种高分辨率数据集上进行了训练和测试

---

## Modifications

- 我在 options/train 下添加了配置文件 UHD-Blur.yml
- 我训练完后的模型在 pretrain_model 文件夹下
- 为了测试 UHD 图像，我添加了滑动窗口测试代码 test_UHD.py   
- 为在 GoPro 和 UHD-Blur 上训练和测试修改并添加了相应的脚本

---

## Dependencies

- Python
- Pytorch (1.11)
- scikit-image
- opencv-python
- Tensorboard
- einops

---

## Datasets
请自行下载 GoPro 和 UHD-Blur 等数据集，并在相应代码中指定路径

---

## Train

### GoPro
在 GoPro 上训练 
bash train_GoPro.sh

### UHD-Blur
在 UHD-Blur 上训练 
bash train_UHD-Blur.sh

---

## Test

### GoPro
在 GoPro 上测试 
bash test_GoPro.sh

### UHD-Blur
在 UHD-Blur 上测试
bash test_UHD-Blur.sh

---

## Results

### GoPro
在 GoPro 上测试结果如下：
avg_ssim:0.969231
avg_psnr:34.213578

达到了论文中汇报的水平

### UHD-Blur
在 UHD-Blur 上测试结果如下：
avg_ssim:0.896829
avg_psnr:31.817174

该方法在 UHD 图像上表现也较好，但由于显存等限制只能使用滑动窗口（test_UHD.py）来测试，因此效果和性能上会有些不足


