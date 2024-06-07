import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from models import build_model
from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
from video_dataset.video_dataset import VideoDataSet
from video_dataset.dataset_config import get_dataset_config
from opts import arg_parser

import matplotlib.pyplot as plt
import torchvision
import cv2

def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result

def create_model(args):

    cudnn.benchmark = False
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5
    elif args.modality == 'gray':
        args.input_channels = 1

    model, arch_name = build_model(args, test_mode=True)


    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint)
    else:
        print("=> creating model '{}'".format(arch_name))

    return model

    result = model(data) 
def test_videos(videos, model, args):

    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    scale_size = args.input_size
   
    augmentor = transforms.Compose( [
            GroupCenterCrop(512),
            GroupScale(scale_size),
            GroupCenterCrop(scale_size),
            Stack(threed_data=args.threed_data),
            ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
            GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
            ]
        )

    model = model.cuda()
    model.eval()


    label ={ 0: 'Incompelement_Penetration', 1: 'Normal_Penetration', 2: 'Over_Penetration', 3: 'unkown', -1 : 'unkown'}

    for video in videos:
        f =  open(video, 'rb') 

        output_name = os.path.join('images', os.path.basename(video.replace('.raw', '.yuv')))


        image_stack = []
        i = 0
        predicted = -1
        while True:
            bytes = f.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            i += 1
            # if i % 4 == 0:
            #     image_stack.append(img)
            image_stack.append(img)
            if len(image_stack) == 8:

                imgs = augmentor(image_stack)
                imgs.unsqueeze_(0)

                imgs = imgs.cuda()
                output = eval_a_batch(imgs, model, args.num_clips, args.num_crops, args.threed_data)
                _, predicted = torch.max(output.data, 1)

                predicted = predicted.cpu().numpy()[0]
                print(video, predicted)
                
                image_stack.clear()
            
            # Convert PIL Image to OpenCV format
            img_cv = np.array(img)

            # Write predicted label on the image
            cv2.putText(img_cv, label[predicted], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            # cv2.imwrite('images/output.png', img_cv)          
            img_np = np.array(img_cv)

            # Write the numpy array to a .raw file
            with open(output_name, 'ab') as raw_file:
                img_np.tofile(raw_file)        
        cmd = f"ffmpeg -y -s 640x512 -pix_fmt gray -r 30 -i {output_name} -c:v libx264 \
                -pix_fmt yuv420p {output_name.replace('yuv', 'mp4')}; rm {output_name}"
        os.system(cmd)
        f.close()
    

if __name__ == '__main__':

    global args
    parser = arg_parser()
    args = parser.parse_args()

    args.num_classes = 4
    
    # 遍历‘data/test_2500’中的文件， 找到*.raw文件， 输出该文件所在完整路径
    v_path = os.path.join('data', 'test_2500')
    
    videos = []
    # 遍历images目录中的所有文件
    for root, dirs, files in os.walk(v_path):
        for file in files:
            if file.endswith('.raw'):
                videos.append(os.path.join(root, file))
    for line in videos:
        print(line)
    
    model = create_model(args)


    test_videos(videos, model, args)
    