
import random
import os

def split_trainval(f_pene, ext):
    # 判断 f_pene 的类型， 是字符串，还是list
    lines = []
    if isinstance(f_pene, str):
        f = open(f_pene, 'r')
        lines = f.readlines()
        f.close()
    elif isinstance(f_pene, list):
        lines = f_pene

    samples = dict()
    Small_Penetration = list()
    for line in lines:
        line = line.strip()
        if "Small_Penetration" in line:
            Small_Penetration.append(line)
            continue
        
        items = line.split(' ')
        if items[-1] in samples.keys():
            samples[items[-1]].append(line)
        else:
            samples[items[-1]] = [line]
    
    for key in samples.keys():
        print(key, len(samples[key]))

    # 随机打乱
    train = []
    val = []
    val_num = 30
    for key in samples.keys():
        num = len(samples[key])
        random.shuffle(samples[key])            # 随机选取，测试用例在训练集中全覆盖， 样本覆盖不全，效果会差（说明训练素材量还不足）
        train += samples[key][:(num-val_num)]
        val += samples[key][(num-val_num):]
    
    with open(f"images/train_{ext}.txt", 'w') as f:
        random.shuffle(train)                   # 随机打乱
        for item in train:
            # 字符串中去掉开头的 'images/' 
            item = item[7:]
            f.write("%s\n" % item)
    with open(f"images/val_{ext}.txt", 'w') as f:
        for item in val:
            # 字符串中去掉开头的 'images/'
            item = item[7:]
            f.write("%s\n" % item)

    with open(f"images/val_Small_Penetration.txt", 'w') as f:
        for item in Small_Penetration:
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
        elif 'Small_Penetration' in line:
            line = line[7:] # 去掉开头的 'images/'


            items = line.split(' ')
            path = items[0]
            depth =  -1 * int(items[-1])


            items = path.split('/')[0].split('_')
            thick = 2500
            if len(items) == 2:
                thick = int(items[-1])

            if depth < thick - 250:
                result.append(line)
        else:
            line = line[7:] # 去掉开头的 'images/'
            result.append(line)
    
    with open(f"images/khd_filter.txt", 'w') as f:
        for item in result:
            f.write("%s\n" % item)
    
    return result

def split_trainval_random(lines, ext):
    # 遍历 lines 中所有的行， 判断图片文件是否存在， 如果不存在， 则删除这一行
    new_lines = []
    for line in lines:
        line = line.strip()
        items = line.split(' ')
        start = int(items[1])
        path = os.path.join('images', items[0], f'{start:05d}.png')
        if os.path.exists(path):
            new_lines.append(line)
    print(f"valid {len(new_lines)} lines", f" in Total {len(lines)} lines")
    # lines 随机打乱， 前80%作为训练集， 后20%作为验证集
    random.shuffle(new_lines)
    ratio = 0.9
    train = lines[:int(len(lines)*ratio)]
    val = lines[int(len(lines)*ratio):]
    with open(f"images/train_{ext}.txt", 'w') as f:
        for item in train:
            f.write("%s\n" % item)

    with open(f"images/val_{ext}.txt", 'w') as f:
        for item in val:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # 遍历images中所有文件，找到所有KHD.txt 文件
    # 将所有KHD.txt文件行合并

    all_khd = get_all_lines('images', 'KHD.txt')
    all_pene = get_all_lines('images', 'penetration.txt')
    split_trainval(all_pene, 'penetration')

    all_khd_filter = filter_khd_info(all_khd)

    split_trainval_random(all_khd_filter, 'depth')
