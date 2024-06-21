

# 训练与测试脚本

## 激光焊接熔透分析

* 训练脚本：train_3d.sh
* 测试脚本：test_3d.sh
* 视频分析脚本：video_demo.sh



## 激光焊接熔深分析

* 训练脚本：train_3d_depth.sh
* 测试脚本：test_3d_depth.sh
* 视频分析脚本：video_demo_depth.sh


## 激光焊接熔深/熔透同时测试

* video_demo_both.sh
  注意，该脚本涉及两个模型，因而pretrained设置是无效的，需要video_demo.py内部修改模型地址
