#coding=utf-8

import os
from PIL import Image
from torch.utils.data import Dataset

class WeldingDataset(Dataset):
    """Class for getting data as a Dict
    Args:

    Output:
        sample : Dict of images and labels"""

    def __init__(self, file, train=True, transform=None):
        # 读取图像分类的数据集, 格式如下
        # xxxxx.jpg 0
        # yyyyy.jpg 1

        self.file = file
        self.train = train
        self.num = 0
        self.data = []
        self.label = []
        self.load_data()
        self.transform = transform

    def load_data(self):
        with open(self.file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(line[0])
                self.label.append(int(line[1]))
                self.num += 1
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        img = self.transform(img)
        label = self.label[idx]
        # print(torch.max(img), torch.min(img))
        return img, label
    
if __name__ == '__main__':
    print("hello world")