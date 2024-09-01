python3 test_3d.py --type regression --temporal_module_name TSN  --backbone_net resnet --depth 50  --without_t_stride --criterion MSE \
    --datadir images/ --dataset laser_welding_depth  --modality gray  --norm -10000 \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ -b 1 -j 1  --num_clips 1 --num_crops 1 --disable_scaleup  \
    --pretrained snapshots/laser_welding_depth-gray-TSN-b3-sum-resnet-50-f8-multisteps-bs16-e150_20240808_114820/checkpoint.pth.tar \
