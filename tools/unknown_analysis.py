
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 必须包含一个参数，即视频文件的路径
    parser.add_argument('--file', type=str, default='snapshots/laser_welding-gray-TAM-b3-sum-resnet-18-f8/test_1crops_4clips_224.csv')
    args = parser.parse_args()

    
    unknown_file = 'images/unknown.txt'
    with open(unknown_file, 'r') as f:
        unknown_list = f.readlines()
        unknown_list = [x.strip() for x in unknown_list]
    
    # 读取csv文件
    with open(args.file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    
    print('unknown_list:', len(unknown_list), 'predict_list:', len(lines))

    # 拼接 unknown_list 和 predict_list 同一行内容， 输出到文件
    with open('images/unknown_predict.txt', 'w') as f:
        for i in range(len(unknown_list)):
            f.write(unknown_list[i] + ' ' + lines[i] + '\n')



