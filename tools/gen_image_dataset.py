import os
from PIL import Image
import pandas as pd
import argparse
import concurrent.futures
import numpy as np
def extract_image(video_path, image_path):
    with open(video_path, 'rb') as video:
        i = 0
        while True:
            bytes = video.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            img.save(f'{image_path}/{i:05d}.png')
            i += 1

def extract_images(v_lists, image_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(extract_image, v_lists, image_paths)

def get_image_max_mean(image_path):
    img = Image.open(image_path)
    img = img.convert('L')

    # 获取灰度图像最亮的100个点的平均值
    img_data = np.array(img).flatten()
    top_100_data = np.sort(img_data)[-100:]
    return np.mean(top_100_data[:-10])

def gen_depth_labels(v_lists, image_paths, output, param:dict):

    depth = param['thickness']
    frames = param['frames']
    speed = param['speed']
    fps = param['fps']
    
    khd_name = output + "/KHD.txt"
    pene_name = output + "/penetration.txt"
    f_khd = open(khd_name, 'w')
    f_pene = open(pene_name, 'w')
    
    for i in range(len(v_lists)):
        # 提取v文件所在的目录
        path = os.path.dirname(v_lists[i])
        csv_file = ''
        # 读取path文件夹下，以 "_KHD.csv"结尾的文件
        for file in os.listdir(path):
            if file.endswith('_KHD.csv'):
                csv_file = os.path.join(path, file)
        # 读取CSV文件
        df = pd.read_csv(csv_file, usecols=[0, 1])
        # 截取第一列数据，数值范围在1000-8300之间的数据
        df = df[(df['Keyhole Depth X (um)'] >= 500) & (df['Keyhole Depth X (um)'] <= 990000)]
        
        start = int(fps * 0.5) # 0.5s * 200fps
        step = 32
        end = frames # 999
        
        for j in range(start, end - step, step):
            s = j / fps * speed * 1000.0
            e = (j + step) / fps * speed * 1000.0
            vals =  df[(df['Keyhole Depth X (um)'] >= s) & (df['Keyhole Depth X (um)'] <= e)]
            
            img_path = os.path.join(image_paths[i], f'{(j + step):05d}.png')
            max_mean = get_image_max_mean(img_path)

            if vals.empty:
                print(path, j, s, e, "empty")
                continue 
            elif max_mean < 25:
                continue                
            else:                

                value = int(vals.iloc[:,1].min())
                # if value < depth + 250:
                line = f'{image_paths[i]} {j} {j + step} {value}\n'
                f_khd.write(line)


                ## 标签判断
                if 'Incomplete_Penetration' in v_lists[i]:
                    if(value > depth): #未熔透
                        pene_label = 0
                    else:
                        pene_label = value                        
                elif 'Small_Penetration' in v_lists[i]:
                    if value > depth + 250:
                        pene_label = 0
                    elif value < depth - 250:
                        pene_label = 1
                    else:
                        pene_label = value # 保留原值
                elif 'Normal_Penetration' in v_lists[i]: # 熔透与过熔透
                    if value < depth:     # 熔透状态
                        pene_label = 1
                    else:
                        pene_label = value
                elif "Over_Penetration" in v_lists[i]:
                    if value < depth:     # 熔透状态
                        pene_label = 2    # 过熔透状态
                    else:
                        pene_label = value
                    
                line = f'{image_paths[i]} {j} {j + step} {pene_label}\n'
                f_pene.write(line)

    f_khd.close()
    f_pene.close()



def gen_mix_labels(v_lists, image_paths, output, param:dict):

    depth = param['thickness']
    frames = param['frames']
    speed = param['speed']
    fps = param['fps']
    
    mix_name = output + "/mix.txt"
    f_mix = open(mix_name, 'w')
    
    for i in range(len(v_lists)):
        # 提取v文件所在的目录
        path = os.path.dirname(v_lists[i])
        csv_file = ''
        # 读取path文件夹下，以 "_KHD.csv"结尾的文件
        for file in os.listdir(path):
            if file.endswith('_KHD.csv'):
                csv_file = os.path.join(path, file)
        # 读取CSV文件
        df = pd.read_csv(csv_file, usecols=[0, 1])
        # 截取第一列数据，数值范围在1000-8300之间的数据
        df = df[(df['Keyhole Depth X (um)'] >= 500) & (df['Keyhole Depth X (um)'] <= 990000)]
        
        start = int(fps * 0.5) # 0.5s * 200fps
        step = 32
        end = frames # 999

        
        for j in range(start, end - step, step):
            s = j / fps * speed * 1000.0
            e = (j + step) / fps * speed * 1000.0
            vals =  df[(df['Keyhole Depth X (um)'] >= s) & (df['Keyhole Depth X (um)'] <= e)]

            img_path = os.path.join(image_paths[i], f'{(j + step):05d}.png')
            max_mean = get_image_max_mean(img_path)

            if vals.empty:
                print(path, j, s, e, "empty")
                continue    
            elif max_mean < 25:
                continue            
            else:                

                value = int(vals.iloc[:,1].min())

                ## 标签判断
                if 'Incomplete_Penetration' in v_lists[i]:
                    if(value > depth): #未熔透
                        pene_label = 0
                    else:
                        pene_label = -1                        
                elif 'Small_Penetration' in v_lists[i]:
                    if value > depth + 250:
                        pene_label = 0
                    elif value < depth - 250:
                        pene_label = 1
                    else:
                        pene_label = -1 # 保留原值
                elif 'Normal_Penetration' in v_lists[i]: # 熔透与过熔透
                    if value < depth:     # 熔透状态
                        pene_label = 1
                    else:
                        pene_label = -1
                elif "Over_Penetration" in v_lists[i]:
                    if value < depth:     # 熔透状态
                        pene_label = 2    # 过熔透状态
                    else:
                        pene_label = -1
                
                stable = 0
                # if value < depth + 250:
                line = f'{image_paths[i]} {j} {j + step} {pene_label} {value} {stable}\n'
                f_mix.write(line)
    f_mix.close()


def gen_stable_labels(v_lists, image_paths, output, param:dict,label_file:str):
    stable_dict = dict()
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            items = line.split(',')
            key = items[0]
            surface = items[1]
            if key not in stable_dict:
                stable_dict[key] = dict()
            stable_dict[key][surface] = items[2]
    # print(stable_dict)

    frames = param['frames'] 
    fps = param['fps']
    
    stable_name = output + "/stable.txt"
    f_stable = open(stable_name, 'w')
    
    
    for i in range(len(v_lists)):
        # 提取v文件所在的目录
        path = os.path.dirname(v_lists[i])
        path = os.path.basename(path)
        if '_' in path:
            path = str(int(path.split('_')[1]))
        key = path + '#'
        
        if key not in stable_dict:
            print(f'{key} not in stable_dict')
            continue
        face = stable_dict[key]

        top = face['1']
        bottom = face['2']

        start = int(fps * 0.5) # 0.5s * 200fps
        step = 32
        end = frames # 999
        
        for j in range(start, end - step, step):           
            line = f'{image_paths[i]} {j} {j + step} {top} {bottom}\n'
            f_stable.write(line)

    f_stable.close()


if __name__ == "__main__":
    # 需要解析图片， 则加上 --image 参数
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--image', action='store_true', default=False, help='extract images from video files')
    args = argparse.parse_args()
    

    data_list = [
                 ['20240603_2500', 16.667 , 1000], 
                 ['20240605_1900', 16.667 , 1000], 
                 ['20240605_2500', 16.667 , 1200], 
                 ['20240722_2500', 16.667 , 1200], 
                 ['20240722_3500_MAG', 25.0 , 1200], 
                 ['20240722_4500_MAG', 25.0 , 1200], 
                 ['20240722_5500', 8.333 , 1200], 
                 ['20240722_5500_MAG', 25.0, 1200], 
                 ['20240722_7500_MAG', 16.667 , 1500], 
                ]
    for data in data_list:
        dir, speed, frames = data

        items = dir.split('_')
        thick = 2500
        if len(items) >= 2:
            thick = int(dir.split('_')[1])
        
        param = dict()
        param['thickness'] = -1 * thick
        param['frames'] = frames - 200
        param['speed'] = speed
        param['fps'] = 200
        
        with open('data/{}/raw.txt'.format(dir), 'r') as f:
            lines = f.readlines()

        v_lists = []
        for line in lines:
            line = line.strip()
            line = os.path.normpath(line)
            video_file = os.path.join('data', dir, line)            
            v_lists.append(video_file)
        
        output = f'images/{dir}'
        if not os.path.exists(output):
            os.makedirs(output)

        image_paths = []
        for video_file in v_lists:
            bold_class_id_str = video_file.split('/')[-3]
            sample_id_str = video_file.split('/')[-2]
            image_path = os.path.join(output, bold_class_id_str)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            
            image_path = os.path.join(output, bold_class_id_str, sample_id_str)
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            image_paths.append(image_path)

        if args.image:
            extract_images(v_lists, image_paths)
        
        gen_depth_labels(v_lists, image_paths, output, param)
        gen_mix_labels(v_lists, image_paths, output, param)
        gen_stable_labels(v_lists, image_paths, output, param, 'data/stable/stable_label_all.txt')




