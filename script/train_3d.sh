



python train_3d.py --temporal_module_name TAM  --backbone_net resnet --depth 18 --without_t_stride \
    --datadir images/ --dataset laser_welding  --modality gray \
    --augmentor_ver v3 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.01  -b 16 -j 8 --epochs 50 --lr_scheduler multisteps 
