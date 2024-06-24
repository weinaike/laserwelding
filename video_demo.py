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
from utils.utils import build_dataflow, AverageMeter, accuracy, get_augmentor
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

def create_model(args, num_class):
    args.num_classes = num_class
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


def test_cls(videos, model, args):
    
    model = model.cuda()
    model.eval()

    mean = model.mean(args.modality)
    std = model.std(args.modality)

    augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                  std=std, disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)


    label ={ 0: 'Incompelement_Penetration', 1: 'Normal_Penetration', 2: 'Over_Penetration', 3: 'black', -1 : 'unkown'}
    if args.type == 'stable':
        label = { 0: 'stable', 1: 'unstable', -1 : 'unkown'}

    for video in videos:
        f =  open(video, 'rb') 
        f_res = open(os.path.join('result', os.path.basename(video.replace('.raw', '.txt'))), 'w')
        output_name = os.path.join('result', os.path.basename(video.replace('.raw', '.yuv')))


        image_stack = []
        i = 0
        predicted = -1
        while True:
            bytes = f.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            i += 1
            if i % args.frames_per_group == 0:
                image_stack.append(img)

            
            
            if len(image_stack) == args.groups:

                imgs = augmentor(image_stack)
                imgs.unsqueeze_(0)

                imgs = imgs.cuda()
                output = eval_a_batch(imgs, model, args.num_clips, args.num_crops, args.threed_data)
                _, predicted = torch.max(output.data, 1)

                predicted = predicted.cpu().numpy()[0]
                print(video, i, predicted)
                
                image_stack.clear()

            f_res.write("{} {}\n".format(i, predicted))
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
        f_res.close()



def test_depth(videos, model, args):

    
    model = model.cuda()
    model.eval()

    mean = model.mean(args.modality)
    std = model.std(args.modality)
    augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                std=std, disable_scaleup=args.disable_scaleup,
                                threed_data=args.threed_data,
                                is_flow=True if args.modality == 'flow' else False,
                                version=args.augmentor_ver)


    for video in videos:
        f =  open(video, 'rb') 

        f_res = open(os.path.join('result', os.path.basename(video.replace('.raw', '_depth.txt'))), 'w')
        output_name = os.path.join('result', os.path.basename(video.replace('.raw', '_depth.yuv')))

        image_stack = []
        i = 0
        predicted = 0
        while True:
            bytes = f.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            i += 1
            if i % args.frames_per_group == 0:
                image_stack.append(img)

            if len(image_stack) == args.groups:

                imgs = augmentor(image_stack)
                imgs.unsqueeze_(0)

                imgs = imgs.cuda()
                output = eval_a_batch(imgs, model, args.num_clips, args.num_crops, args.threed_data)

                predicted = output.cpu().numpy()[0][0] * -3000.0
                print(video, predicted)
                
                image_stack.clear()
            
            f_res.write("{} {}\n".format(i, predicted))


            # Convert PIL Image to OpenCV format
            img_cv = np.array(img)

            # Write predicted label on the image
            cv2.putText(img_cv, str(predicted), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            # cv2.imwrite('images/output.png', img_cv)          
            img_np = np.array(img_cv)

            # Write the numpy array to a .raw file
            with open(output_name, 'ab') as raw_file:
                img_np.tofile(raw_file)        
        cmd = f"ffmpeg -y -s 640x512 -pix_fmt gray -r 30 -i {output_name} -c:v libx264 \
                -pix_fmt yuv420p {output_name.replace('yuv', 'mp4')}; rm {output_name}"
        os.system(cmd)
        f.close()
        f_res.close()


def test_both(videos, model_cls, model_depth, args):
    
    model_cls = model_cls.cuda()
    model_cls.eval()


    model_depth = model_depth.cuda()
    model_depth.eval()


    mean = model_cls.mean(args.modality)
    std = model_cls.std(args.modality)
    augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                std=std, disable_scaleup=args.disable_scaleup,
                                threed_data=args.threed_data,
                                is_flow=True if args.modality == 'flow' else False,
                                version=args.augmentor_ver)
    label ={ 0: 'Incompelement_Penetration', 1: 'Normal_Penetration', 2: 'Over_Penetration', 3: 'black', -1 : 'unkown'}
    for video in videos:
        f =  open(video, 'rb') 

        f_res = open(os.path.join('result', os.path.basename(video.replace('.raw', '_both.txt'))), 'w')
        output_name = os.path.join('result', os.path.basename(video.replace('.raw', '_both.yuv')))

        image_stack = []
        i = 0
        predicted = 0
        depth = 0
        while True:
            bytes = f.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            i += 1
            if i % args.frames_per_group == 0:
                image_stack.append(img)

            if len(image_stack) == args.groups:

                imgs = augmentor(image_stack)
                imgs.unsqueeze_(0)

                imgs = imgs.cuda()
                output = eval_a_batch(imgs, model_cls, args.num_clips, args.num_crops, args.threed_data)
                _, predicted = torch.max(output.data, 1)

                predicted = predicted.cpu().numpy()[0]

                if predicted == 0:
                    output = eval_a_batch(imgs, model_depth, args.num_clips, args.num_crops, args.threed_data)
                    depth = output.cpu().numpy()[0][0] * -3000.0
                else:
                    depth = 0
                print(video, predicted, depth)
                
                image_stack.clear()
            
            f_res.write("{} {} {}\n".format(i, predicted, depth))


            # Convert PIL Image to OpenCV format
            img_cv = np.array(img)

            # Write predicted label on the image
            cv2.putText(img_cv, f'{label[predicted]} / {depth}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            # cv2.imwrite('images/output.png', img_cv)          
            img_np = np.array(img_cv)

            # Write the numpy array to a .raw file
            with open(output_name, 'ab') as raw_file:
                img_np.tofile(raw_file)        
        cmd = f"ffmpeg -y -s 640x512 -pix_fmt gray -r 30 -i {output_name} -c:v libx264 \
                -pix_fmt yuv420p {output_name.replace('yuv', 'mp4')}; rm {output_name}"
        os.system(cmd)
        f.close()
        f_res.close()

if __name__ == '__main__':

    parser = arg_parser()
    args = parser.parse_args()
      
    videos = []

    # 遍历‘data/test_2500’中的文件， 找到*.raw文件， 输出该文件所在完整路径
    v_path = os.path.join('data', 'test_2500')
    if args.type == 'stable':
        v_path = os.path.join('data', 'stable_20240621.txt')
        
    if os.path.isdir(v_path):
        # 遍历images目录中的所有文件
        for root, dirs, files in os.walk(v_path):
            for file in files:
                if file.endswith('.raw'):
                    videos.append(os.path.join(root, file))
    else:
        with open(v_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split(' ')
            if items[1] != 'unknown':
                continue
            if items[0].endswith('.raw'):
                videos.append(os.path.join('data', items[0]))

    for line in videos:
        print(line)
    
    

    if args.type == 'class':
        model = create_model(args, 4)
        test_cls(videos, model, args)
    elif args.type == 'regression':
        model = create_model(args, 1)
        test_depth(videos, model, args)
    elif args.type == 'both':
        args.pretrained = 'snapshots/laser_welding-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_161355/checkpoint.pth.tar'
        model_cls = create_model(args, 4)
        args.pretrained = 'snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_161701/checkpoint.pth.tar'
        model_depth = create_model(args, 1)
        test_both(videos, model_cls, model_depth, args)

    else:# for stable
        model = create_model(args, 2)
        test_cls(videos, model, args)
    