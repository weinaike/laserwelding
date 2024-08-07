import shutil
import os
import time
import multiprocessing
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 2)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def distance(output, target, norm = -2500.0):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        res = torch.mean(torch.abs((output - target)* norm))
        return res

def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        elif version == 'v3':
            augments += [
                GroupRandomScale(scale_range),
                GroupCenterCrop(image_size),
            ]
        elif version == 'v4':
            augments += [
                GroupScale(image_size),
                GroupCenterCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        if version == 'v3':
            augments += [
                GroupScale(image_size),
                GroupCenterCrop(image_size),
            ]
        elif version == 'v4':
            augments += [
                GroupScale(image_size),
                GroupCenterCrop(image_size),
            ]
        else:
            scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
            augments += [
                GroupScale(scaled_size),
                GroupCenterCrop(image_size)
            ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0, norm = 0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images)
            target = target.cuda(gpu_id, non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec2 = accuracy(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec2)
                prec1 /= world_size
                prec2 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top2.update(prec2[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top2=top2), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top2.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None, norm = 0):
    print('validate')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_labels = []          

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            # measure accuracy and record loss
            prec1, prec2 = accuracy(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec2)
                prec1 /= world_size
                prec2 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top2.update(prec2[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    cm = confusion_matrix(all_labels, all_preds)
    # 输出三分类测试混淆矩阵
    print('Confusion Matrix:')
    print(cm)
    return top1.avg, top2.avg, losses.avg, batch_time.avg, cm





def train_regression(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0, norm = -10000.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0

    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images)
            

            output = output.view(-1)
            # output = torch.sigmoid(output).view(-1)
            target = target.cuda(gpu_id, non_blocking=True)
            target = target / norm

            loss = criterion(output, target)
            # loss = criterion(output/target, target/target) 
            # print('loss:', loss.item(), 'output:', output/target, 'target:', target/target)

            # measure accuracy and record loss
            prec1 = distance(output, target, norm)
            prec5 = prec1 # 无效

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate_regression(data_loader, model, criterion, gpu_id=None, norm = -10000.0):
    print('validate_regression')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_labels = []    

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            
            # # 将images 转为numpy，取最后一张图，保存为文件
            # save = images.cpu().numpy()
            # save = save[-1]
            # np.save('save_train.npy', save)

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            
            # compute output
            output = model(images)
            
            output = output.view(-1)
            # output = torch.sigmoid(output).view(-1)
            target = target.cuda(gpu_id, non_blocking=True)

            target = target / norm

            loss = criterion(output, target) 
            # print(output, target, loss.item())
            # loss = criterion(output/target, target/target) 

            # measure accuracy and record loss
            prec1 = distance(output, target, norm)
            # print('prec1:', prec1)
            prec5 = prec1 # 无效

            if dist.is_initialized():
                # print("dist.is_initialized()")
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)


            all_preds.extend(output.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            # print('avg:', losses.avg, 'count:', losses.count)
    
    
    diff = np.abs((np.array(all_labels) - np.array(all_preds)) * norm)
    cm = [np.max(diff) , np.min(diff), np.sum(diff < 1000), len(all_labels)] 
    # 距离统计
    print('max , min,  right, total:')
    print(cm)
    return top1.avg, top5.avg, losses.avg, batch_time.avg, cm



def train_mix(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0, norm = -10000.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0

    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            out1, out2, out3 = model(images)
            target = target.cuda(gpu_id, non_blocking=True)

            bs, _ = target.size()
            valid1 = []
            valid2 = []
            for i in range(bs):
                if int(target[i,0])>= 0:
                    valid1.append(i)
                if int(target[i,0]) == 0:
                    valid2.append(i)
            
            loss1 = 0
            prec2 = 0
            if len(valid1) > 0:
                # 分类 
                loss1 = criterion[0](out1[valid1], target[valid1,0].long())
                prec1, _ = accuracy(out1[valid1], target[valid1,0].long())
            
            # 回归
            loss2 = 0
            prec2 = 0
            if len(valid2) > 0:
                out2 = out2.view(-1)
                target[valid2,1] = target[valid2,1] / norm
                loss2 = criterion[1](out2[valid2], target[valid2,1])
                prec2 = distance(out2[valid2], target[valid2,1], norm)

            # 分类
            loss3 = 0
            prec3 = 0
            # loss3 = criterion[0](out3, target[:,2])
            # prec3, _ = accuracy(out3, target[:,2])
            
            
            loss = loss1 + 4 * loss2 #+ loss3
            if loss > 0:
                loss.backward()

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec2)
                dist.all_reduce(prec3)
                prec1 /= world_size
                prec2 /= world_size
                prec3 /= world_size

            losses.update(loss.item(), images.size(0))
            if len(valid1) > 0:
                top1.update(prec1[0], len(valid1))
            if len(valid2) > 0:                
                top2.update(prec2, len(valid2))
            
            top3.update(prec3, images.size(0))
            # compute gradient and do SGD step
            

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'prec1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'dist2 {top2.val:.3f} ({top2.avg:.3f})\t'
                      'prec3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top2=top2, top3=top3), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break


    return top1.avg, top2.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate_mix(data_loader, model, criterion, gpu_id=None, norm = -10000.0):
    print('validate_mix')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    status_all_preds = []
    status_all_labels = []    
    depth_all_preds = []
    depth_all_labels = []    
    stable_all_preds = []
    stable_all_labels = []

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            
            # # 将images 转为numpy，取最后一张图，保存为文件
            # save = images.cpu().numpy()
            # save = save[-1]
            # np.save('save_train.npy', save)

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            
            out1, out2, out3 = model(images)
            target = target.cuda(gpu_id, non_blocking=True)

            bs, _ = target.size()
            valid1 = []
            valid2 = []
            for i in range(bs):
                if int(target[i,0]) >= 0:
                    valid1.append(i)
                if int(target[i,0]) == 0:
                    valid2.append(i)

            loss1 = 0
            prec1 = 0
            if len(valid1) > 0:
                # 分类
                
                loss1 = criterion[0](out1[valid1], target[valid1,0].long())
                prec1, _ = accuracy(out1[valid1], target[valid1,0].long())
                _, predicted = torch.max(out1.data, 1)
               
            
            # 回归
            loss2 = 0
            prec2 = 0
            if len(valid2) > 0:
                out2 = out2.view(-1)
                target[valid2,1] = target[valid2,1] / norm
                loss2 = criterion[1](out2[valid2], target[valid2,1])
                prec2 = distance(out2[valid2], target[valid2,1], norm)

            # 分类
            loss3 = 0
            prec3 = 0
            # loss3 = criterion[0](out3, target[:,2])
            # prec3, _ = accuracy(out3, target[:,2])
            
            loss = loss1 + 4 * loss2 #+ loss3

            if dist.is_initialized():
                # print("dist.is_initialized()")
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec2)
                dist.all_reduce(prec3)
                prec1 /= world_size
                prec2 /= world_size
                prec3 /= world_size
            losses.update(loss.item(), images.size(0))
            if len(valid1) > 0:
                top1.update(prec1[0], len(valid1))
            if len(valid2) > 0:
                top2.update(prec2, len(valid2))
            top3.update(prec3, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

            if len(valid1) > 0:
                status_all_preds.extend(predicted[valid1].cpu().numpy())
                status_all_labels.extend(target[valid1,0].cpu().numpy())
            if len(valid2) > 0 :            
                depth_all_preds.extend(out2[valid2].cpu().numpy())
                depth_all_labels.extend(target[valid2,1].cpu().numpy())
            # stable_all_preds.extend(out3.cpu().numpy())
            # stable_all_labels.extend(target[:,2].cpu().numpy())
            
            # print('avg:', losses.avg, 'count:', losses.count)
    cms = []
    if len(status_all_labels) > 0:
        # float to int
        # print(status_all_labels)
        # print(status_all_preds)
        status_all_labels = [int(i) for i in status_all_labels]
        status_all_preds = [int(i) for i in status_all_preds]
        cm1 = confusion_matrix(status_all_labels, status_all_preds)
        # 输出三分类测试混淆矩阵
        print('Confusion Matrix:')
        print(cm1)    
        cms.append(cm1)
    if len(depth_all_labels) > 0:
        diff = np.abs((np.array(depth_all_labels) - np.array(depth_all_preds)) * norm)
        cm2 = [np.max(diff) , np.min(diff), np.sum(diff < 1000), len(depth_all_labels)] 
        # 距离统计
        print('max , min,  right, total:')
        print(cm2)
        cms.append(cm2)

    # cm3 = confusion_matrix(stable_all_labels, stable_all_preds)
    # # 输出三分类测试混淆矩阵
    # print('Confusion Matrix:')
    # print(cm3)   
    return top1.avg, top2.avg, losses.avg, batch_time.avg, cms
