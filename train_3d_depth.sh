python train_3d.py --temporal_module_name TAM  --backbone_net resnet --depth 18 --without_t_stride --criterion MSE \
    --datadir images/ --dataset laser_welding_depth  --modality gray --augmentor_ver v4 --frames_per_group 1 --groups 8 --threed_data \
    --logdir snapshots/ --lr 0.01  -b 16 -j 8 --epochs 50 --lr_scheduler multisteps
