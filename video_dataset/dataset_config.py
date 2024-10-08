

DATASET_CONFIG = {
    'st2stv2': {
        'num_classes': 174,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'mini_st2stv2': {
        'num_classes': 87,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 3
    },
    'kinetics400': {
        'num_classes': 400,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'test_list_name': 'test.txt',
        'filename_seperator': ",",
        'image_tmpl': '{:05d}.bmp',
        'filter_video': 30
    },
    'mini_kinetics400': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'test_list_name': 'mini_test.txt',
        'filename_seperator': ";",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 30
    },
    'moments': {
        'num_classes': 339,
        'train_list_name': 'train.txt',
        'val_list_name': 'val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'mini_moments': {
        'num_classes': 200,
        'train_list_name': 'mini_train.txt',
        'val_list_name': 'mini_val.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.jpg',
        'filter_video': 0
    },
    'laser_welding': {
        'num_classes': 4,
        'train_list_name': 'train_penetration.txt',
        'val_list_name': 'val_penetration.txt',
        'test_list_name': 'unknown.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.png',
        'filter_video': 0
    },
    'laser_welding_depth': {
        'num_classes': 1,
        'train_list_name': 'train_depth.txt',
        'val_list_name': 'val_depth.txt',
        'test_list_name': 'val_depth.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.png',
        'filter_video': 0
    },
    'laser_welding_stable': {
        'num_classes': 2,
        'train_list_name': 'train_stable.txt',
        'val_list_name': 'val_stable.txt',
        'test_list_name': 'val_stable.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.png',
        'filter_video': 0
    },
    'laser_welding_all': {
        'num_classes': 8,
        'train_list_name': 'train_mix.txt',
        'val_list_name': 'val_mix.txt',
        'test_list_name': 'test_mix.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:05d}.png',
        'filter_video': 0
    },    
}


def get_dataset_config(dataset):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name']
    val_list_name = ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
