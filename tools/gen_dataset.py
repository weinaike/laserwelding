
import random
import os

def split_trainval_count(lines, ext, val_num = 30):  
    random.seed(42)
    samples = dict()
    unknown = list()
    for line in lines:
        line = line.strip()       
        items = line.split(' ')

        key = int(items[-1])
        if key < 0:
            unknown.append(line)
            continue

        if key in samples.keys():
            samples[key].append(line)
        else:
            samples[key] = [line]
    
    for key in samples.keys():
        print(key, len(samples[key]))

    # 随机打乱
    train = []
    val = []
    # val_num = 30
    for key in samples.keys():
        num = len(samples[key])
        random.shuffle(samples[key])            # 随机选取，测试用例在训练集中全覆盖， 样本覆盖不全，效果会差（说明训练素材量还不足）
        train += samples[key][:(num-val_num)]
        val += samples[key][(num-val_num):]
    
    with open(f"images/train_{ext}.txt", 'w') as f:
        for item in train:
            # 字符串中去掉开头的 'images/' 
            item = item[7:]
            f.write("%s\n" % item)
    with open(f"images/val_{ext}.txt", 'w') as f:
        for item in val:
            # 字符串中去掉开头的 'images/'
            item = item[7:]
            f.write("%s\n" % item)

    with open(f"images/unknown.txt", 'w') as f:
        for item in unknown:
            # 字符串中去掉开头的 'images/'
            item = item[7:]
            f.write("%s\n" % item)


def get_all_lines(dir, name = 'KHD.txt'):
    all_lines = []
    # 遍历images目录中的所有文件
    for root, dirs, files in os.walk(dir):
        for file in files:
            # 如果文件是一个KHD.txt文件
            if file==name:
                # 打开文件并读取所有的行
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                # 将这些行添加到all_lines列表中
                all_lines.extend(lines)

    return all_lines


def filter_khd_info(lines):

    with open(f"images/khd_all.txt", 'w') as f:
        for item in lines:
            f.write("%s\n" % item.strip()) 

    result = []
    for line in lines:
        line = line.strip()
        if 'Over_Penetration' in line or 'Normal_Penetration' in line:
            continue
        # elif 'Small_Penetration' in line:
        else:
            # line = line[7:] # 去掉开头的 'images/'

            items = line.split(' ')
            path = items[0]
            depth =  -1 * int(items[-1])


            items = path.split('/')[1].split('_')
            thick = 2500
            if len(items) >= 2:
                thick = int(items[1]) # 从文件夹名获取厚度

            if depth < thick - 250:
                result.append(line)
    
    with open(f"images/khd_filter.txt", 'w') as f:
        for item in result:
            f.write("%s\n" % item)
    
    return result


def split_trainval_ratio(lines, ext, val_ratio = 0.1):
    random.seed(42)
    # 遍历 lines 中所有的行， 判断图片文件是否存在， 如果不存在， 则删除这一行
    new_lines = []
    for line in lines:
        line = line.strip()
        items = line.split(' ')
        start = int(items[1])
        path = os.path.join(items[0], f'{start:05d}.png')
        
        if os.path.exists(path):
            new_lines.append(line)
        else:
            print(f"remove {line},  {path} not exist")
    print(f"valid {len(new_lines)} lines", f" in Total {len(lines)} lines")
    # lines 随机打乱， 前80%作为训练集， 后20%作为验证集
    
    random.shuffle(new_lines)

    ratio = 1 - val_ratio
    train = new_lines[:int(len(new_lines)*ratio)]
    val = new_lines[int(len(new_lines)*ratio):]

    train.sort()
    val.sort()
    with open(f"images/train_{ext}.txt", 'w') as f:
        for item in train:
            f.write("%s\n" % item[7:])

    with open(f"images/val_{ext}.txt", 'w') as f:
        for item in val:
            f.write("%s\n" % item[7:])


def get_black_lines(root, dir, single = True):
    path = os.path.join(root, dir)
    samples = os.listdir(path)

    black_list = []

    for sample in samples:
        img_dir = os.path.join(path, sample)
        images = os.listdir(img_dir)
        images.sort()
        start = int(images[0].split('.')[0])
        end = int(images[-1].split('.')[0])

        step = 32
        if(end - start < step):
            continue

        for i in range(start, end - step, step//4):
            if single:
                black_list.append(f"{img_dir} {i} {i + step} {3}")
            else:
                black_list.append(f"{img_dir} {i} {i + step} {3} {-1} {-1}")
                
    return black_list
        
if __name__ == "__main__":
    # 遍历images中所有文件，找到所有KHD.txt 文件
    # 将所有KHD.txt文件行合并
    # black_list = get_black_lines('images', 'black_sample')
    all_khd = get_all_lines('images', 'KHD.txt')
    all_pene = get_all_lines('images', 'penetration.txt')
    all_stable = get_all_lines('images', 'stable.txt')
    all_mix = get_all_lines('images', 'mix.txt')
    black_list = get_black_lines('images', 'black_sample')
    
    black_list_3label = get_black_lines('images', 'black_sample', False)
    split_trainval_ratio(all_mix + black_list_3label, 'mix', val_ratio = 0.1)


    # split_trainval_count(all_pene + black_list, 'penetration', val_num = 50)

    # all_khd_filter = filter_khd_info(all_khd)
    # split_trainval_ratio(all_khd_filter, 'depth', val_ratio = 0.1)


    split_trainval_ratio(all_stable, 'stable', val_ratio = 0.1)
