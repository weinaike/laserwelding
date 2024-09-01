# 一、安装

1. 安装依赖库

   ```
   pip install -r requirements.txt
   ```
2. 需要运行video_demo.py，还需要安装ffmepg，保存视频

   ```
   sudo apt install ffmpeg
   ```
3. 项目目录结构

>   .
>    ├── data				# 存放原始数据
>    ├── images				# 存放
>    ├── models				# 网络模型
>    ├── README.md			# 说明文档
>    ├── result				# 测试运行结果
>    ├── script				# 训练与测试脚本
>    ├── snapshots			# 训练运行结果
>    ├── tools				# 分析工具
>    ├── video_dataset		# 数据集

# 二、准备数据集

## 1. 原始数据处理

    * 原始数据存放到data目录下
    * 数据按如下格式存放
      ├── 20240603_2500                 	# 归类文件夹名： 格式要求：日期_试样厚度
      │   └──  Incomplete_Penetration   	# 存放未熔透数据
      │     └── 2                       			# 样本ID
      │         ├── 112002_2#.raw       	# 视频文件
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_FWS.csv       #LDD标定文件
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_FWS.png
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_KHD.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_KH.png
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TBC.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TBW.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TLE.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TP.png
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TP_Point_Cloud.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_TRE.csv
      │         ├── 3mm-PASS-S-0000000367-D-20240603-T-112002_WH.png
      │         └── 3mm-PASS-S-0000000367-D-20240603-T-112002_WPH.csv
      │   ├── Normal_Penetration        	# 存放正常熔透数据
      │   ├── Over_Penetration          	# 存放过熔透数据
      │   ├── Small_Penetration         	# 存放临界熔透数据
      │   └── raw.txt                   		# 本目录下所有.raw文件列表

## 2. 制作数据集与标签

    运行script/gen_image_labels.py 生成数据和标签，默认仅生成标签， 若需提取图像， 增加参数--image

    ```python
    # 需要解析图片(从raw视频生成png图片)， 并生成标签
    python tools/gen_image_labels.py --image

    # 仅生成标签
    python tools/gen_image_labels.py
    ```

## 3. 生成数据集

    运行script/gen_dataset.py 生成数据集，

    生成的数据集存放在images目录下， 包含提取的帧文件，与生成的数据集文件
    熔透状态数据集：train_penetration.txt, val_penetration.txt
    熔深数据集：train_depth.txt, val_depth.txt
    稳定性数据集：train_stable.txt, val_stable.txt

    这些数据会在video_dataset/dataset_config.py中配置使用

# 三、训练与测试

每个任务都各自训练模型， 各任务的训练与测试方法如下：（执行命令前， cd到项目根目录下）

## 1. 激光焊接熔透分析

* 训练脚本：script/train_3d.sh
* 测试脚本：script/test_3d.sh
* 视频分析脚本：script/video_demo.sh

  * 根据加载的模型不同， 可以计算稳定性预测或者熔深状态预测
  * 注意:
    * 稳定性预测， 需要修改video_demo.py中v_path路径， 提供视频稳定性标注文件
    * 熔深预测， 需要修改video_demo.py中v_path路径， 待测试的样本路径

## 2. 激光焊接熔深分析

* 训练脚本：script/train_3d_depth.sh
* 测试脚本：script/test_3d_depth.sh
* 视频分析脚本：script/video_demo_depth.sh

## 3. 激光焊接稳定性分析

* 训练脚本：script/train_3d_stable.sh
* 测试脚本：script/test_3d_stable.sh
* 视频分析脚本：script/video_demo_stable.sh

## 4. 激光焊接熔深/熔透/稳定性同时测试

* video_demo_both.sh
  注意，该脚本涉及多个模型，因而pretrained设置是无效的，需要video_demo.py内部修改模型地址

```
  elif args.type == 'both':
          args.pretrained = 'snapshots/laser_welding-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e120_20240805_163936/checkpoint.pth.tar'
          args.depth = 18
          model_cls = create_model(args, 4)
          args.pretrained = 'snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-50-f8-multisteps-bs16-e150_20240806_213424/checkpoint.pth.tar'
          args.depth = 50
          model_depth = create_model(args, 1)

          #stable
          args.pretrained = 'snapshots/laser_welding_stable-gray-resnet-18-f8-multisteps-bs16-e80_20240819_112343/checkpoint.pth.tar'
          args.temporal_module_name = 'TSN'
          args.depth = 18
          model_stable = create_model(args, 2)
          test_both(videos, model_cls, model_depth, args, model_stable)  
```

# 四、数据分析

批量测试，可运行如下脚本， 在result文件夹中生成测试视频与图表

```
sh script/runtest.sh
```

具体的分析内容，请查看研究报告
