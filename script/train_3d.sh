



python train_3d.py --temporal_module_name TAM  --backbone_net resnet --depth 34 --without_t_stride \
    --datadir images/ --dataset laser_welding  --modality gray \
    --augmentor_ver v4 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.001  -b 16 -j 8 --epochs 120 --lr_scheduler multisteps --lr_steps 80 100 
