




python3 video_demo.py --type both --temporal_module_name TSN  --backbone_net resnet  --depth 50  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0 --augmentor_ver v4  --frames_per_group 1 --groups 8 --disable_scaleup --norm -10000  \
    --pretrained snapshots/laser_welding_depth-gray-TSN-b3-sum-resnet-50-f8-multisteps-bs16-e150_20240808_114820/checkpoint.pth.tar \

