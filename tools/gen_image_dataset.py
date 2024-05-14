import os


dirs = {'20240507/rt_+3_1000_1.5_3mm' : 0,  '20240507/rt_+3_1000_2_3mm' : 1,  '20240507/wt_+3_1000_3_3mm':2}

f_train = open('train_img.txt', 'w')
f_val = open('val_img.txt', 'w')

for key, val in dirs.items():
    files = os.listdir(key)
    num = len(files)
    for i  in range(num):
        file = files[i]
        file = os.path.join('data', key, file)
        if i < num * 0.8:
            f_train.write(file + ' ' + str(val) + '\n')
        else:
            f_val.write(file + ' ' + str(val) + '\n')