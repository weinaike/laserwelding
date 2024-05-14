import torch
import torchsummary
from models.threed_models.i3d_resnet import I3D_ResNet


model = I3D_ResNet(50, 400, 0.5, without_t_stride=False)
model.to("cpu")
dummy_data = (3, 32, 224, 224)
model.eval()
model_summary = torchsummary.summary(model, input_size=dummy_data, device="cpu")
print(model_summary)



checkpoint = torch.load("zoo/K400-I3D-ResNet-50-f32.pth", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# 给输入输出取个名字
input_names = ["input"]
output_names = ["output"]

# 创建一个虚拟输入
dummy_input = torch.randn(1, 3, 32, 224, 224, device="cpu")

# 将模型导出为 ONNX 格式
torch.onnx.export(
    model,
    dummy_input,
    "K400-I3D-ResNet-50-f32.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
)
