import torch
import torchsummary
import sys
sys.path.insert(0, './')
from models.threed_models.i3d_resnet import I3D_ResNet
from models.twod_models.resnet import resnet


# model = I3D_ResNet(50, 400, 0.5, without_t_stride=False)
model = resnet(18, 3, False, 8, temporal_module_name = None, 
                dw_conv = 'True', blending_frames = 3, blending_method = 'sum', 
                dropout = 0.5, pooling_method ='max', imagenet_pretrained=True)
model.to("cpu")
dummy_data = (24, 224, 224)
model.eval()
model_summary = torchsummary.summary(model, input_size=dummy_data, device="cpu")
print(model_summary)



# checkpoint = torch.load("zoo/K400-I3D-ResNet-50-f32.pth", map_location='cpu')
checkpoint = torch.load("zoo/model_best.pth", map_location='cpu')
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(new_state_dict)

# 给输入输出取个名字
input_names = ["input"]
output_names = ["output"]

# 创建一个虚拟输入
dummy_input = torch.randn(1, 24, 224, 224, device="cpu")

# 将模型导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,
    "resnet.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
)
