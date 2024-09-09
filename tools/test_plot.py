import os
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')



def plot_video_info(CLASS = True, DEPTH = True, STABLE = True):
    # 遍历‘data/test_2500’中的文件， 找到*.raw文件， 输出该文件所在完整路径
    # v_path = os.path.join('data', 'test_2500')
    v_path = os.path.join('data', 'test_20240722')
    
    videos = []
    # 遍历images目录中的所有文件
    for root, dirs, files in os.walk(v_path):
        for file in files:
            if file.endswith('.raw'):
                videos.append(os.path.join(root, file))

    stable_dict = dict()
    with open('data/stable/stable_label_all.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            items = line.split(',')
            key = items[0]
            surface = items[1]
            if key not in stable_dict:
                stable_dict[key] = dict()
            stable_dict[key][surface] = items[2]

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
                front = 0
                back = 0
                if len(item) > 3:
                    front = int(item[3])
                    back = int(item[4])
                if label != 0:
                    depth = 0

                if id < 100:
                    continue
                elif id > 999:
                    continue
                else:
                    result.append([id, label, depth, front, back])
  
        result = np.array(result)
        idx, thick, speed =  dir.split('/')[-1].split('_')
        idx = int(idx)
        thick = -1 * int(thick)
        speed =  1000 / 60 * float(speed) 

        frames = result[:, 0] /200 * speed *1000
        # result : 0: distance, 1: label, 2: depth, 3:stable_front, 4:stable_back, 5: KHD
        # 获取该目录下 KHD.txt后缀文件
        for filename in os.listdir(dir):
            if filename.endswith('KHD.csv'):
                csv_file = os.path.join(dir, filename)
                df = pd.read_csv(csv_file, usecols=[0, 1])
                # print(df.head())
                # 截取第一列数据，数值范围在1000-8300之间的数据
                df = df[(df['Keyhole Depth X (um)'] >= 500) & (df['Keyhole Depth X (um)'] <= 990000)]
                #读取该文件的所有行
                x = df['Keyhole Depth X (um)']
                y = df['Keyhole Depth Z (um)']

                f = interp1d(x, y)
                new_y = f(frames).reshape(-1, 1)
                result = np.concatenate((result, new_y), axis=1)
        
        frames_res = list()
        depth_res = list()
        for x, y in zip(frames, result[:, 2]):
            if y != 0:
                frames_res.append(x)
                depth_res.append(y)    
        
        ldd =  result[:, 5]

        fig, ax1 = plt.subplots(figsize=(12, 6))        
        plt.title(name)
        if DEPTH == True:
            ax1.scatter(frames, ldd, label='LDD KHD Data', color='b', s=1)
            ax1.scatter(frames_res, depth_res, label='Depth Predict', color='g', s=1)        
            # Draw horizontal line at y=1000
            ax1.axhline(y=thick, color='b', linestyle='--', label='Sample Thickness')
            ax1.set_ylabel('Penetration Depth (um)', color='g')
            ax1.set_xlabel('Distance (um)')
            fig.legend(loc='center right')
            plt.tight_layout(rect=[0, 0, 0.8, 1])
            plt.savefig(os.path.join('result', name.replace('.raw', '_depth.png')))
        if CLASS == True:
            ax2 = ax1.twinx()
            ax2.plot(frames, result[:, 1], label='Penetration_Status\n0: Incomplement Penetration\n1: Normal Penetration\n2: Over Penetration\n3: Unkown (No Info) Status', color='r')
            ax2.set_ylim([0, 3])
            plt.yticks(np.arange(-1, 5, 1))
            ax2.tick_params(axis='y', colors='r')
            ax2.set_ylabel('Penetration Status Predict', color='r')
            fig.legend(loc='center right')
            plt.tight_layout(rect=[0, 0, 0.8, 1])
            plt.savefig(os.path.join('result', name.replace('.raw', '.png')))
        if STABLE == True:
            key = name.replace('.raw', '')
            front = int(stable_dict[key]['1'])
            back = int(stable_dict[key]['2'])

            ax2 = ax1.twinx()
            ax2.plot(frames, result[:, 3], label='Front Stable Predict', color='y', linestyle='--')
            ax2.plot(frames, result[:, 4], label='Back Stable Predict', color='y')      
            ax2.set_ylabel('\nStable Predict', color='y')
            ax2.axhline(y=front, color='y', linestyle='-.', label='Front Stbale Label')
            ax2.axhline(y=back, color='y', linestyle=':', label='Back Stable Label')
            # ax2.spines['right'].set_color('y')# 右侧轴
            # 设置刻度线和刻度标签的颜色
            ax2.tick_params(axis='y', colors='y')
            
            fig.legend(loc='center right')
            plt.yticks(np.arange(0, 100, 20))
            plt.tight_layout(rect=[0, 0, 0.8, 1])            
            plt.savefig(os.path.join('result', name.replace('.raw', '_all.png')))    

        plt.close()


def plot_depth_info(model: str):
    csv_file = os.path.join('snapshots', model, 'test_1crops_1clips_224.csv')
    result = []
    with open(csv_file) as f:
        predicts = f.readlines()
        
        for pred in predicts:
            item = pred.strip().split(',')
            label = float(item[0])
            
            depth = float(item[1])
            result.append([label, depth])
    result = np.array(result)
    # 第一列排序result
    sorted_indices = np.argsort(result[:, 0])[::-1]
    result = result[sorted_indices]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('Depth Predict')

    ax1.plot(result[:, 0], label='LDD label Depth', color='b')
    ax1.plot(result[:, 1], label='Predict Depth', color='g')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Penetration Depth (um)')
    ax1.legend()
    plt.savefig(os.path.join('result', 'depth.png'))
    plt.close()



def plot_stable_info(model:str):
    csv_file = os.path.join('snapshots', model, 'test_1crops_1clips_224.csv')
    result = []
    with open(csv_file) as f:
        predicts = f.readlines()
                
        for pred in predicts:

            item = pred.strip().split(',')
            label = float(item[0])
            
            depth = float(item[1])
            result.append([label, depth])
    front = np.array(result[0:-2:2])
    back = np.array(result[1:-1:2])
    # 第一列排序result
    sorted_indices = np.argsort(front[:, 0])[::-1]
    front = front[sorted_indices]
    sorted_indices = np.argsort(back[:, 0])[::-1]
    back = back[sorted_indices]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('Front Stable Predict')

    ax1.plot(front[:, 0], label='Front Stable Label', color='b')
    ax1.plot(front[:, 1], label='Front Stable Predict', color='g')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('score')
    ax1.legend()
    plt.savefig(os.path.join('result', 'stable_front.png'))
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title('back Stable Predict')

    ax1.plot(back[:, 0], label='Back Stable Label', color='b')
    ax1.plot(back[:, 1], label='Back Stable Predict', color='g')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('score')
    ax1.legend()
    plt.savefig(os.path.join('result', 'stable_back.png'))
    plt.close()


if __name__ == '__main__':
    plot_video_info(CLASS = False, DEPTH = False, STABLE = True)
    plot_depth_info(model='laser_welding_depth-gray-TSN-b3-sum-resnet-50-f8')
    plot_stable_info(model='laser_welding_stable-gray-resnet-18-f8')


