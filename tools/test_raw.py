import os
from PIL import Image
import argparse
import concurrent.futures

def extract_image(video_path, image_path):
    with open(video_path, 'rb') as video:
        i = 0
        while True:
            bytes = video.read(640*512)
            if not bytes:
                break
            img = Image.frombytes('L', (640, 512), bytes)
            #获取最大像素值
            # max_pixel = img.getextrema()[1]
            # print(max_pixel)
            # 归一化
            # img = img.point(lambda x: x*255//max_pixel)
            
            img.save(f'{image_path}/{i:05d}.png')
            i += 1

def extract_images(v_lists, image_paths):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(extract_image, v_lists, image_paths)


if __name__ == "__main__":
    # 需要解析图片， 则加上 --image 参数
    argparse = argparse.ArgumentParser()

    v_lists = ['data/test_20240722/Normal_Penetration/64/64#.raw']
    image_paths = ['zoo/64/']
    
    
    extract_images(v_lists, image_paths)
    