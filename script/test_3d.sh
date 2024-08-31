




python3 test_3d.py --temporal_module_name TAM  --backbone_net resnet  --depth 18  --without_t_stride \
    -e --datadir images/ --dataset laser_welding  --modality gray --augmentor_ver v4 --frames_per_group 4  --dense_sampling  --groups 8  \
    --logdir snapshots/ -b 1 -j 1  --num_clips 1 --num_crops 1 --disable_scaleup  \
    --pretrained snapshots/laser_welding-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e120_20240805_163936/checkpoint.pth.tar