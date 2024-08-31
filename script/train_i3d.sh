    
python train_3d.py --backbone_net i3d_resnet --depth 18 --without_t_stride \
    --datadir images/ --dataset laser_welding  --modality gray --threed_data  \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.001  -b 16 -j 8 --epochs 80 --lr_scheduler multisteps --lr_steps 50 70 


    #  --pretrained snapshots/laser_welding_depth-gray-TAM-b3-sum-resnet-50-f8-multisteps-bs16-e120_20240730_140309/checkpoint.pth.tar 


    #  --no_imagenet_pretrained
