import os
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt



def plot_video_info():
    # 遍历‘data/test_2500’中的文件， 找到*.raw文件， 输出该文件所在完整路径
    v_path = os.path.join('data', 'test_2500')
    
    videos = []
    # 遍历images目录中的所有文件
    for root, dirs, files in os.walk(v_path):
        for file in files:
            if file.endswith('.raw'):
                videos.append(os.path.join(root, file))
    for line in videos:
        dir = os.path.dirname(line)
        name = os.path.basename(line)

        result = []
        with open(os.path.join('result', name.replace('.raw', '_both.txt'))) as f:
            predicts = f.readlines()
            for pred in predicts:
                item = pred.strip().split(' ')
                id = int(item[0])
                label = int(item[1])
                depth = float(item[2])
                if id < 100:
                    continue
                elif id > 999:
                    continue
                else:
                    result.append([id, label, depth])      
        result = np.array(result)

        frames = result[:, 0] /200 *16.6667 *1000
        # result : 0: distance, 1: label, 2: depth, 3: KHD
        # 获取该目录下 KHD.txt后缀文件
        for filename in os.listdir(dir):
            if filename.endswith('KHD.csv'):
                csv_file = os.path.join(dir, filename)
                df = pd.read_csv(csv_file, usecols=[0, 1])
                # print(df.head())
                # 截取第一列数据，数值范围在1000-8300之间的数据
                df = df[(df['Keyhole Depth X (um)'] >= 500) & (df['Keyhole Depth X (um)'] <= 90000)]
                #读取该文件的所有行
                x = df['Keyhole Depth X (um)']
                y = df['Keyhole Depth Z (um)']

                f = interp1d(x, y)
                new_y = f(frames).reshape(-1, 1)
                result = np.concatenate((result, new_y), axis=1)
        
        
        fig, ax1 = plt.subplots(figsize=(10, 6))        
        plt.title(name)
        ax1.plot(frames, result[:, 3], label='LDD KHD Data', color='b', linestyle=':')
        ax1.plot(frames, result[:, 2], label='Depth Predict', color='g')        
        ax1.set_ylabel('Penetration Depth', color='g')
        ax1.set_xlabel('Distance')
        ax2 = ax1.twinx()
        ax2.plot(frames, result[:, 1], label='Penetration_Status\n0: Incomplement Penetration\n1: Normal Penetration\n2: Over Penetration\n3: Unkown (No Info) Status', color='r')
        ax2.set_ylim([0, 3])
        plt.yticks(np.arange(-1, 5, 1))
        ax2.set_ylabel('Penetration Status Predict', color='r')
        fig.legend()
        plt.savefig(os.path.join('result', name.replace('.raw', '.png')))
        plt.close()




def plot_depth_info():
    csv_file = os.path.join('snapshots', 'laser_welding_depth-gray-TAM-b3-sum-resnet-18-f8', 'test_1crops_1clips_224.csv')
    result = []
    with open(csv_file) as f:
        predicts = f.readlines()
        
        for pred in predicts:
            item = pred.strip().split(',')
            label = float(item[0])
            
            depth = float(item[1])
            result.append([label, depth])
    result = np.array(result)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('Depth Predict')
    ax1.plot(result[:, 0], label='LDD Data', color='b')
    ax1.plot(result[:, 1], label='Depth Predict', color='g')
    ax1.set_ylabel('Penetration Depth', color='g')
    ax1.legend()
    plt.savefig(os.path.join('result', 'depth.png'))
    plt.close()


def plot_stable(x, fws, tre, tle, lx, predicts, name, status=-1):
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(name.replace('.txt', ' Stable Status') )
    ax1.plot(x, fws, label='LDD FWS', color='g')
    ax1.plot(x, tre, label='LDD TRE', color='r')
    ax1.plot(x, tle, label='LDD TLE', color='b')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('um')
    ax2 = ax1.twinx()
    ax2.plot(lx, predicts, label='Stable Status\n-1: unknown\n 0: stable\n 1: unstable', color='y')
    ax2.set_ylabel('Stable Status', color='y')
    ax2.set_ylim([-1, 2])
    plt.yticks(np.arange(-1, 2, 1))


    fig.legend()
    plt.savefig(os.path.join('result', name.replace('.txt', f'_stable_{status}.png')))
    plt.close()


                          
def plot_stable_info():
    v_path = os.path.join('data', 'stable_20240621.txt')

    videos = dict()
    knows = dict()
    with open(v_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        items = line.split(' ')
        if items[1] != 'unknown':
            dir = items[0].split('/')[-2]
            if items[1] == 'stable':
                knows[dir] = 0
            else:
                knows[dir] = 1            
        else:
            name = os.path.basename(items[0].replace('.raw', '.txt'))
            dir = items[0].split('/')[-2]
            videos[name] = dir

    for name, dir in videos.items():
        result = os.path.join('result', name)
        ldd = os.path.join('data', 'stable',  f'{dir}_all.csv')
        if not os.path.exists(ldd):
            continue

        df = pd.read_csv(ldd)
        x = df['x']
        fws = df['FWS']
        tre = df['TRE']
        tle = df['TLE']

        id = []
        predicts = []
        with open(result) as f:
            ress = f.readlines()
            for res in ress:
                item = res.strip().split(' ')
                id.append(int(item[0]))
                predicts.append(int(item[1]))
                
        id = np.array(id)
        predicts = np.array(predicts)

        lx = id /200 *16.6667 *1000
        plot_stable(x, fws, tre, tle, lx, predicts, name)


    for dir , status in knows.items():
        ldd = os.path.join('data', 'stable',  f'{dir}_all.csv')
        if not os.path.exists(ldd):
            continue
        df = pd.read_csv(ldd)
        x = df['x']
        fws = df['FWS']
        tre = df['TRE']
        tle = df['TLE']

        lx = np.array([0,90000]) 
        predicts = np.array([int(status), int(status)])

        plot_stable(x, fws, tre, tle, lx, predicts, f'{dir}.txt', status=status)

if __name__ == '__main__':
    # plot_video_info()
    # plot_depth_info()
    plot_stable_info()


