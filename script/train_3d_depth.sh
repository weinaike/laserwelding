python train_3d.py --temporal_module_name TAM  --backbone_net resnet --depth 34 --without_t_stride --criterion MSE \
    --datadir images/ --dataset laser_welding_depth  --modality gray --norm -10000 --opti adam \
    --augmentor_ver v3 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.0001  -b 16 -j 8 --epochs 150 --lr_scheduler multisteps --lr_steps 120 
    
    #  --pretrained snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-50-f8-multisteps-bs16-e120_20240730_140309/checkpoint.pth.tar 


    #  --no_imagenet_pretrained
