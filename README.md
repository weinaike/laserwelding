# 训练与测试脚本

## 激光焊接熔透分析

* 训练脚本：script/train_3d.sh
* 测试脚本：script/test_3d.sh
* 视频分析脚本：script/video_demo.sh
  * 根据加载的模型不同， 可以计算稳定性预测或者熔深状态预测
  * 注意稳定性预测， 需要修改video_demo.py中v_path路径， 提供视频稳定性标注文件
  * 注意熔深预测， 需要修改video_demo.py中v_path路径， 待测试的样本路径

## 激光焊接熔深分析

* 训练脚本：script/train_3d_depth.sh
* 测试脚本：script/test_3d_depth.sh
* 视频分析脚本：script/video_demo_depth.sh

## 激光焊接熔深/熔透同时测试

* video_demo_both.sh
  注意，该脚本涉及两个模型，因而pretrained设置是无效的，需要video_demo.py内部修改模型地址

具体数据分析工具， 详见tools/readme.md
