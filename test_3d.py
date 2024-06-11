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


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = False

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)

    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes = num_classes
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5
    elif args.modality == 'gray':
        args.input_channels = 1

    model, arch_name = build_model(args, test_mode=True)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be one.")
        elif args.modality == 'gray':
            if len(args.mean) != 1:
                raise ValueError("When training with gray, dim of mean must be one.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be one.")
        elif args.modality == 'gray':
            if len(args.std) != 1:
                raise ValueError("When training with gray, dim of std must be one.")
        std = args.std

    model = model.cuda()
    model.eval()

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint, strict=True)
    else:
        print("=> creating model '{}'".format(arch_name))

    model = torch.nn.DataParallel(model).cuda()

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                  std=std, disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)

    # Data loading code
    data_list = os.path.join(args.datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}".format(args.num_clips))

    val_dataset = VideoDataSet(args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=not args.evaluate,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        top1 = AverageMeter()
        top5 = AverageMeter()
    else:
        logfile = open(os.path.join(log_folder, 'test_{}crops_{}clips_{}.csv'.format(
            args.num_crops, args.num_clips, args.input_size)), 'w')

    total_outputs = 0
    outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
    # switch to evaluate mode
    model.eval()
    total_batches = len(data_loader)

    all_preds = []
    all_labels = []          

    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            # save = video.cpu().numpy()
            # save = save[-1]
            # np.save('save_test.npy', save)
            output = eval_a_batch(video, model, num_clips=args.num_clips, num_crops=args.num_crops,
                                  threed_data=args.threed_data)

            if args.evaluate:
                label = label.cuda(non_blocking=True)
                _, predicted = torch.max(output.data, 1)
                # measure accuracy
                prec1, prec5 = accuracy(output, label, topk=(1, 2))
                top1.update(prec1[0], video.size(0))
                top5.update(prec5[0], video.size(0))
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

                if args.debug:
                    # 显示 video 中的所有图片，绘制到一张图上， video.Size([1, 12, 224, 224]) 
                    show = video.view(3, -1, 224, 224)
                    # show = video.view(-1, 1, 224, 224)
                    # for i in range(show.size(0)):
                    #     # 保存（1，224，224）致文件
                    #     torch.save(show[i], f'snapshots/test{i}.pth')
                        
                    show = show.transpose(0, 1)
                    grid =  torchvision.utils.make_grid(show, nrow=4)
                    grid = grid.numpy().transpose((1, 2, 0))
                    plt.figure()
                    plt.imshow(grid)
                    plt.title(f"Label: {label[0]}, predicted: {predicted[0]}")
                    plt.show()

            else:
                # testing, store output to prepare csv file
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output
                
                predictions = np.argsort(output, axis=1)
                if args.type == 'regression':
                    # print("{},{},{}".format(str(label.numpy()), str(output),  str(output * -3000.0), file=logfile))
                    all_preds.extend(output[0] * -3000.0)
                    all_labels.append(label.numpy())
                    print("{},{}".format(str(label.numpy()[0]), output[0][0] * -3000.0), file=logfile)
                else:
                    for ii in range(len(predictions)):
                        # preds = [id_to_label[str(pred)] for pred in predictions[ii][::-1][:5]]
                        temp = predictions[ii][::-1][:2]
                        preds = [str(pred) for pred in temp]
                        print("{},{}".format(str(label[ii].numpy()), ",".join(preds)), file=logfile)
            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        outputs = outputs[:total_outputs]
        print("Predict {} videos.".format(total_outputs), flush=True)
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details.npy'.format(
            "val" if args.evaluate else "test", args.num_crops,
            args.num_clips, args.input_size)), outputs)

    if args.evaluate:
        print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@2: {:.4f}'.format(
            args.input_size, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg),
            flush=True)
        print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@2: {:.4f}'.format(
            args.input_size, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg),
            flush=True, file=logfile)
        
        cm = confusion_matrix(all_labels, all_preds)
        # 输出三分类测试混淆矩阵
        print('Confusion Matrix:')
        print(cm)
        print(cm, file=logfile)
    # print(all_labels, all_preds)
    if args.type == 'regression':
        print("avg depth:", np.mean(np.abs(np.array(all_labels).reshape(-1) - np.array(all_preds).reshape(-1))))
    logfile.close()

if __name__ == '__main__':
    main()
