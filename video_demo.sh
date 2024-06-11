




python3 video_demo.py --type class --temporal_module_name TAM  --backbone_net resnet  --depth 18  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0  --frames_per_group 1 --groups 8 --augmentor_ver v4 --disable_scaleup \
    --pretrained snapshots/laser_welding-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_161355/checkpoint.pth.tar \

