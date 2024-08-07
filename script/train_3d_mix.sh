python train_3d.py --temporal_module_name TAM  --backbone_net resnet_multihead --depth 34 --without_t_stride --criterion MIX \
    --datadir images/ --dataset laser_welding_all  --modality gray --norm -10000 --opti adam \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.001  -b 16 -j 8 --epochs 150 --lr_scheduler multisteps --lr_steps 120


    #  --pretrained snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-18-f8-multisteps-bs16-e50_20240610_145410/checkpoint.pth.tar \
    #  --no_imagenet_pretrained
