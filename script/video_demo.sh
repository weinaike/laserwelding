




python3 video_demo.py --type class --temporal_module_name TSN  --backbone_net resnet  --depth 18  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0  --frames_per_group 1 --groups 8 --augmentor_ver v4 --disable_scaleup --norm -10000  \
    --pretrained snapshots/laser_welding-gray-TSN-b3-sum-resnet-18-f8-multisteps-bs16-e120_20240808_114843/checkpoint.pth.tar \

