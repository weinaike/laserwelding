import os
from PIL import Image, ImageStat
import numpy as np
import argparse

def calc_image_light_mean(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    stat = ImageStat.Stat(img)
    mean = stat.mean[0]
    return mean

def calc_image_light_max(image_path):
    img = Image.open(image_path)
    img = img.convert('L')

    # 灰度图像最亮的100个点的平均值
    img_data = np.array(img).flatten()
    brightest_points = np.partition(-img_data, 100)[:100]
    mean_of_brightest = np.mean(brightest_points)
    
    return mean_of_brightest



def get_light_info(lines):

    result = []
    count = 0
    for line in lines:
        count += 1
        if count > 100:
            pass # continue            

        line = line.strip()
        items = line.split(' ')
        path = items[0]
        start = int(items[1])
        end = int(items[2])
        label = int(items[3])
        

        all_mean = []
        all_max = []        
        for i in range(start, end):
            image_file = os.path.join('images', path , f'{i:05d}.png')
            try:
                all_mean.append(calc_image_light_mean(image_file))
                all_max.append(calc_image_light_max(image_file))
        
            except Exception as e:
                print(e)                
                continue
        if len(all_mean) == 0:
            continue
        mean_top_10 = sorted(all_mean, reverse=True)[:10]
        max_top_10 = sorted(all_max, reverse=True)[:10]

        result.append((label, sum(all_mean)/len(all_mean), sum(mean_top_10)/len(mean_top_10), 
                       max(all_max), sum(all_max)/len(all_max), sum(max_top_10)/len(max_top_10),
                       path))
    
    # result 重新排序 以label降序排序
    result.sort(key=lambda x: x[0], reverse=True)
  
    np_res = np.array(result)

    return np_res

def light_analysis(file):     
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt

    # 拟合 light_info[:, 1] 和 light_info[:, 2] 之间的关系
    light_info = np.load(file)
    shape = light_info.shape
    for i in range(1, shape[1]-1):
        light_info = np.load(file)
        X = light_info[:, 0].astype(float)
        y = light_info[:, i].astype(float)

        # x， y 以y从小到大排序
        index = np.argsort(y, axis=0)
        X = X[index]
        y = y[index]    
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split( y, X, test_size=0.2, random_state=42)
        # model = LinearRegression()
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(mean_squared_error(y_test, y_pred))
        print(model.score(X_test, y_test))
        print(model.named_steps['linearregression'].coef_)
        print(model.named_steps['linearregression'].intercept_)
        plt.figure()
        plt.scatter(X_test, y_test, color='black')
        plt.scatter(X_test, y_pred, color='blue')
        # 保存图片
        plt.savefig(f'images/light_info_{i}.png')
        plt.close()


if __name__ == "__main__":

    # 需要解析图片， 则加上 --image 参数
    # 需要生成训练和验证文件，则加上 --trainval 参数
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--numpy', action='store_true', default=False, help='extract images from video files')
    args = argparse.parse_args()    

    if args.numpy:    
        file = os.path.join('images', 'khd_filter.txt')
        lines = []
        with open(file, 'r') as f:
            lines = f.readlines()
        
        light_info = get_light_info(lines)
        print(light_info.shape)
        np.save('images/light_info.npy', light_info)
       
    light_analysis('images/light_info.npy')





