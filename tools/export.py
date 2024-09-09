import torch
import torchsummary
import sys
sys.path.insert(0, './')
from models.twod_models.resnet import resnet

def pth2onnx(pth, output, depth=18, cls=4, TMN='TSN'):

    c = 8
    h = 224
    w = 224

    model = resnet(depth, cls, False, c, temporal_module_name = TMN, 
                    dw_conv = 'True', blending_frames = 3, blending_method = 'sum', 
                    dropout = 0.5, pooling_method ='max', imagenet_pretrained=False, modality='gray')
    model.to("cpu")
    dummy_data = (c, h, w)
    model.eval()
    model_summary = torchsummary.summary(model, input_size=dummy_data, device="cpu")
    # print(model_summary)

    checkpoint = torch.load(pth, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(new_state_dict)

    # 给输入输出取个名字
    input_names = ["input"]
    output_names = ["output"]

    # 创建一个虚拟输入
    dummy_input = torch.randn(1, c, h, w, device="cpu")

    # 将模型导出为 ONNX 格式
    torch.onnx.export(
        model,
        dummy_input,
        output,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=False
    )


if __name__ == '__main__':

    pth = ['laser_welding-gray-TSN-b3-sum-resnet-18-f8-multisteps-bs16-e120_20240808_114843', 
            'laser_welding_depth-gray-TSN-b3-sum-resnet-50-f8-multisteps-bs16-e150_20240808_114820',
            'laser_welding_stable-gray-resnet-18-f8-multisteps-bs16-e80_20240819_112343']
    for p in pth:
        task = p.split('-')[0]
        if task == 'laser_welding':
            depth = 18
            cls = 4
        elif task == 'laser_welding_depth':
            depth = 50
            cls = 1
        elif task == 'laser_welding_stable':
            depth = 18
            cls = 2
        pth2onnx(pth=f"snapshots/{p}/checkpoint.pth.tar", output=f'zoo/{task}.onnx', depth=depth, cls=cls)
