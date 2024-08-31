python3 test_3d.py --type regression --backbone_net resnet --depth 18  --without_t_stride --criterion MSE \
    --datadir images/ --dataset laser_welding_stable  --modality gray  --norm 100 \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ -b 1 -j 1  --num_clips 1 --num_crops 1 --disable_scaleup  \
    --pretrained snapshots/laser_welding_stable-gray-resnet-18-f8-multisteps-bs16-e80_20240819_112343/checkpoint.pth.tar \
