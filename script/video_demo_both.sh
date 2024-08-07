




python3 video_demo.py --type both --temporal_module_name TAM  --backbone_net resnet  --depth 50  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0 --augmentor_ver v4  --frames_per_group 1 --groups 8 --disable_scaleup --norm -10000  \
    --pretrained snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_161701/checkpoint.pth.tar \

