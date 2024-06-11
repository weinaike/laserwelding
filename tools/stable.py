import os
import pandas as pd
import argparse
import numpy as np
from scipy.interpolate import interp1d

def get_csv(path, ext):

    csv_file = ''
    # 读取path文件夹下，以 "_KHD.csv"结尾的文件
    for file in os.listdir(path):
        if file.endswith(ext):
            csv_file = os.path.join(path, file)
    # 读取CSV文件
    df = pd.read_csv(csv_file, usecols=[0, 1])
    # 截取第一列数据，数值范围在1000-8300之间的数据
    first_column_name = df.columns[0]
    second_column_name = df.columns[1]
    df = df[(df[first_column_name] >= 0) & (df[first_column_name] <= 90000)]
    
    new_x = np.linspace(10000, 80000, 1400, endpoint= False )  

    f = interp1d(df[first_column_name], df[second_column_name])

    new_y = f(new_x)

    new_df = pd.DataFrame({
        "x": new_x,
        ext.split('.')[0]: new_y
    })

    return new_df 



def gen_stable_labels(v_lists, image_paths, output, param:dict):

    depth = param['thickness']
    frames = param['frams']
    speed = param['speed']
    fps = param['fps']
    
    dir_name = v_lists[0].split('/')[1]

    short = [['ID','distance', 'max_fws', 'min_fws', 'max_te', 'min_te','A', 'B', 'C']]
    for i in range(len(v_lists)):
        # 提取v文件所在的目录
        path = os.path.dirname(v_lists[i])

        fws = get_csv(path, 'FWS.csv')
        tre = get_csv(path, 'TRE.csv')
        tle = get_csv(path, 'TLE.csv')
        
        col_name = tre.columns[1]
        fws[col_name] = tre[col_name]

        col_name = tle.columns[1]
        fws[col_name] = tle[col_name]

        fws['TRE-TLE'] = fws['TRE'] - fws['TLE']
        # 保存 fws 到csv文件
        name = path.split('/')[-1]
        fws.to_csv(os.path.join('data/stable', name +'_all.csv'), index=False)

        x = fws['x']
        fws_np = fws['FWS']
        te_np = fws['TRE-TLE']

        fws_mean = np.mean(fws_np)
        te_mean = np.mean(te_np)

        
        for i in range(10000, 80000, 10000):
            part = fws[(fws['x'] >= i) & (fws['x'] <= i + 10000)]
            #获取各列的最大值
            max_fws = part['FWS'].max()
            max_te = part['TRE-TLE'].max()
            min_fws = part['FWS'].min()
            min_te = part['TRE-TLE'].min()

            A = np.sqrt((max_fws - fws_mean) ** 2 + (min_fws - fws_mean) ** 2)
            B = np.sqrt((max_te - te_mean) ** 2 + (min_te - te_mean) ** 2)
            C = A/te_mean
            short.append([name,i+5000, max_fws, min_fws, max_te, min_te, A, B, C])
    
    print(len(short))
    with open(f'data/stable/ABC_{dir_name}.csv', 'w') as f:
        for line in short:
            f.write(','.join([str(x) for x in line]) + '\n')
        





if __name__ == "__main__":
    # 需要解析图片， 则加上 --image 参数
    # 需要生成训练和验证文件，则加上 --trainval 参数
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--image', action='store_true', default=False, help='extract images from video files')
    argparse.add_argument('--trainval', action='store_true', default=False, help='generate train and val txt file ')
    args = argparse.parse_args()
    

    data_list = [['20240603_2500', 16.667 ], 
                 ['20240605_1900', 16.667 ], 
                 ['20240605_2500', 16.667 ], 
                ]
    for data in data_list:
        dir, speed = data

        items = dir.split('_')
        thick = 2500
        if len(items) == 2:
            thick = int(dir.split('_')[-1])
        
        param = dict()
        param['thickness'] = -1 * thick
        param['frams'] = 999
        param['speed'] = speed
        param['fps'] = 200
        
        with open('data/{}/raw.txt'.format(dir), 'r') as f:
            lines = f.readlines()

        v_lists = []
        for line in lines:
            line = line.strip()
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
        
        if True:
            gen_stable_labels(v_lists, image_paths, output, param)



