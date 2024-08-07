




python3 video_demo.py --type class --temporal_module_name TAM  --backbone_net resnet  --depth 50  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0  --frames_per_group 1 --groups 8 --augmentor_ver v4 --disable_scaleup --norm -10000  \
    --pretrained snapshots/laser_welding-gray-TAM-b3-sum-resnet-34-f8-multisteps-bs16-e50_20240724_093147/checkpoint.pth.tar \

