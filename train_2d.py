import torch
from models.model_2d import resnet10,resnet18,resnet34
from dataset.WeldingDataset import WeldingDataset
from torchvision import transforms
import time
from torch.utils.data import DataLoader
from utils.meter import AverageMeter, Summary, ProgressMeter
import argparse
import logging
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


 
def train(dataloader, model, loss_fn, optimizer, epoch, device, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch))

    size = len(dataloader.dataset)
    model.train()

    end = time.time()
    for batch, (X, y) in enumerate(dataloader):
        data_time.update(time.time() - end)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        logging.debug("pred per batch:{}\n".format( pred)) 
        loss = loss_fn(pred, y)
        # logging.debug("loss per batch:{}\n".format( loss.item())) 
        acc1 = accuracy(pred, y)
        losses.update(loss.item(), X.size(0))
        top1.update(acc1[0].item(), X.size(0))

        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % 100 == 0:
            progress.display(batch + 1)

        # if batch % 2 == 1:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     logging.info(f"train: loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
 
 
def val(dataloader, model, loss_fn, epoch, device):

    losses = AverageMeter('Loss', ':.6f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter( len(dataloader), [losses, top1],prefix='Val: ')

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    losses.update(test_loss, 1)
    top1.update(correct, 1)
    # progress.display_summary()
    logging.info(f"Val: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    return correct
 
 
def test(dataloader, model, device):
    model.eval()
    num = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in dataloader:
            num = num + 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, predicted = torch.max(pred.data, 1)  # get the index of the max log-probability
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            # 输出三分类测试混淆矩阵

    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)


def main(args):

    Epoches = args.epochs
    Batch_Size = args.batch_size
    gpu = args.gpu
    filename = args.model_file
    LR = args.learning_rate
    pretrained = args.pretrained
    workers = args.workers

    ### point_type, weight_mode
    logging.info("Batch_Size: {}".format(Batch_Size))
    train_file = os.path.join("data", "train_img.txt")
    val_file = os.path.join("data", "val_img.txt")
    training_data = WeldingDataset(train_file, train=True)
    val_data = WeldingDataset(val_file, train=False)

    # 中心裁剪， 缩放， 随机裁剪， 随机翻转， 归一化， toTensor
    train_trans = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_trans = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    training_data = WeldingDataset(train_file, train=True, transform=train_trans)
    val_data = WeldingDataset(val_file, train=False, transform=val_trans)


    train_dataloader = DataLoader(training_data, batch_size=Batch_Size, shuffle=True, persistent_workers = True, prefetch_factor = 4, 
                                  drop_last=True,num_workers=workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False,
                                drop_last=False,num_workers=workers, pin_memory=True)
    test_dataloader = DataLoader(val_data, batch_size=1, shuffle=False,
                                 drop_last=False)
 
    # 3.模型加载, 并对模型进行微调

    net = resnet10(num_classes=3)

    if pretrained is not None:
        dict = torch.load(pretrained)
        net.load_state_dict(dict["state_dict"])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = ('cuda:{}'.format(gpu[0]) if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if len(gpu) > 1:
        device_name = 'cuda'
    logging.info("--------------using {}----------------".format(device_name))
    device = torch.device(device_name)

    # 4.pytorch fine tune 微调(冻结一部分层)。这里是冻结网络前30层参数进行训练。
    net.to(device)

    net = torch.nn.DataParallel(net, device_ids = gpu)
 
    # 5.定义损失函数，以及优化器
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.L1Loss(reduction='sum')
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    best_acc = 0.0
    logging.info("Epoches:  {}".format(Epoches))
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    for epoch in range(Epoches):
        logging.info("learning ratio: {}".format(scheduler.get_last_lr()))
        train(train_dataloader, net, criterion, optimizer, epoch, device, args)
        acc = val(val_dataloader, net, criterion, epoch, device)
       
        scheduler.step()        

        # remember best acc@1 and save checkpoint
        is_best = acc > (best_acc - 1e-5)
        best_acc = max(acc, best_acc)

        state_dict = {'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()
            }
        if is_best:
            torch.save(state_dict, filename)

    test(test_dataloader, net, device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model_file', default="best.pth", type=str, 
                        help='save model file path')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step', default=10, type=int, 
                        help='Learning rate step (default: 10)')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='use pre-trained model')
    parser.add_argument('--gpu', nargs='+', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--log_file', default="file.log", type=str,
                        help='log file path.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s-%(levelname)s %(message)s',level=logging.INFO,
                        handlers=[logging.FileHandler(args.log_file,mode="w"), logging.StreamHandler()])
    logging.info("{}".format(vars(args)))
    main(args)


    