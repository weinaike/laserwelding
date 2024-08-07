




python3 video_demo.py --type mix --temporal_module_name TAM  --backbone_net resnet_multihead  --depth 34  --without_t_stride  \
    --modality gray  --logdir snapshots/ --gpu 0 --augmentor_ver v4  --frames_per_group 1 --groups 8 --disable_scaleup --norm -10000  \
    --pretrained snapshots/laser_welding_all-gray-TAM-b3-sum-resnet-34-f8-multisteps-bs16-e150_20240806_095553/checkpoint.pth.tar \

