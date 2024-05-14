import os

dir_label_dict = {'2000w-0.6-wt_2.8mm_570-1370':0, '600w-0.6-wt_0.5mm_300-1100':0, '1200w-0.6-wt_1.7mm_350-1150':0, '1200w-0.6-rt_330-750':1}



#获取当前目录下的所有二级子目录的名字
def get_file_list(path):
    file_list = []
    depth = 0
    for root, dirs, files in os.walk(path):        
        if depth == 1:
            for dir in dirs:
                file_list.append(dir)
        depth += 1
    return file_list

#获取所有二级子目录的相对路径
def gen_dirs_list(path):
    file_list = []
    depth = 0
    for root, dirs, files in os.walk(path):        
        if depth == 1:
            for dir in dirs:
                file_list.append(os.path.join(root, dir))
        depth += 1
    return file_list




train_list = list()
val_list = list()

for dir in gen_dirs_list('.'):
    label = dir_label_dict.get(dir.split('/')[-1])
    print(label)

    files = os.listdir(dir)
    files.sort()
    num = len(files)

    tarin_count = num - 80

    train_id_start =  int(files[0].split('.')[0])

    val_id_start =  train_id_start + tarin_count

    print(train_id_start, val_id_start)

    for i in range(train_id_start, val_id_start - 32 , 8 ):

        train_list.append((dir, i, i + 32,  label))
    
    for i in range(val_id_start, num + train_id_start - 32, 8 ):
        val_list.append((dir, i, i + 32,  label))

# for dir in gen_dirs_list('.'):
#     label = dir_label_dict.get(dir.split('/')[-1])
#     print(label)

#     files = os.listdir(dir)
#     files.sort()
#     num = len(files)

#     tarin_count = num - 80

#     train_id_start =  int(files[0].split('.')[0])

#     val_id_start =  train_id_start + tarin_count

#     print(train_id_start, val_id_start)

#     train_list.append((dir, train_id_start, val_id_start - 1,  label))

#     val_list.append((dir, val_id_start, train_id_start + num - 1 ,  label))

print(len(train_list))
print(len(val_list))

with open('train.txt', 'w') as f:
    for item in train_list:
        out = item[0] + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3])
        f.write("%s\n" % out)

with open('val.txt', 'w') as f:
    for item in val_list:
        out = item[0] + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3])
        f.write("%s\n" % out)
