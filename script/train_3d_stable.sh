



python train_3d.py --backbone_net resnet --depth 18 --without_t_stride --criterion MSE \
    --datadir images/ --dataset laser_welding_stable  --modality gray --norm 100 --opti sgd  \
    --augmentor_ver v3 --frames_per_group 4 --dense_sampling  --groups 8  \
    --logdir snapshots/ --lr 0.001  -b 16 -j 8 --epochs 80 --lr_scheduler multisteps --lr_steps 50 70 
