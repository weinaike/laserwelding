python train_3d.py --datadir ./data --threed_data \
--dataset laser_welding --frames_per_group 1 --groups 8  \
--logdir snapshots/ --lr 0.01 --backbone_net s3d_resnet --depth 18 -b 8 -j 8 --epochs 5
