python train_3d.py --temporal_module_name TAM  --backbone_net resnet --depth 34 --without_t_stride --criterion L1 \
    --datadir images/ --dataset laser_welding_depth  --modality gray \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.001  -b 16 -j 8 --epochs 120 --lr_scheduler multisteps --lr_steps 80 100 \
    #  --pretrained snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_145410/checkpoint.pth.tar \
    #  --no_imagenet_pretrained
