# 原始数据处理与数据集构建

## 提取图像与标注信息

gen_image_labels.py函数作用：

将原始raw视频，解析成图片存储，同时构建大量视频片段作为数据样本，并未每个样本生成对应的数据标签

```python
# 需要解析图片(从raw视频生成png图片)， 则加上 --image 参数
python tools/gen_image_labels.py --image 

# 内部主要执行的是gen_depth_labels函数
```

该函数使用， 根据目标文件目录不同，

```
# 原始目录  
    data_list = [['20240603_2500', 16.667 ], #文件夹名称， 焊接速度m/s
                 ['20240605_1900', 16.667 ], 
                 ['20240605_2500', 16.667 ], 
                ]
#生成文件
images/20240603_2500/KHD.txt
images/20240603_2500/penetration.txt

```

解析完图片后，会根据ldd标注文件， 在各自目录下生成KHD.txt 、penetration.txt文件，

## 构建数据集

gen_dataset.py 根据gen_image_labels.py生成的标签文件， 生成训练集和测试集

```
# 生成熔透分类数据集标签
split_trainval(all_pene + black_list, 'penetration')
#生成的文件，在images目录下 train_penetration.txt val_penetration.txt

# 生成熔深回归数据集标签
split_trainval_random(all_khd_filter, 'depth')
#生成的文件，在images目录下 train_depth.txt val_depth.txt

```

---

# 数据分析

## 测试可视化

test_plot.py 测试结果绘图，当前支持三种

1. 测试视频的预测数据与LDD数据绘制到同一张图中，plot_video_info
2. 将验证集样本结果与LDD标签结果对比，plot_depth_info
3. 绘制稳定状态与标定数据，plot_stable_info

---

# 其他工具

## 导出onnx模型：

export.py 将pytorch模型文件转为onnx文件，以便于部署使用

