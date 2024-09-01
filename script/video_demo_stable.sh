




python3 video_demo.py --type stable --temporal_module_name TSN  --backbone_net resnet  --depth 18  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0  --frames_per_group 4 --groups 8 --augmentor_ver v4 --disable_scaleup --norm 100  \
    --pretrained snapshots/laser_welding_stable-gray-resnet-18-f8-multisteps-bs16-e80_20240819_112343/checkpoint.pth.tar \

