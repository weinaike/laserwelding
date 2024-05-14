python3 test.py --datadir ./data --threed_data \
--dataset kinetics400 --frames_per_group 8 --groups 32 \
--logdir snapshots/  --backbone_net i3d_resnet -d 50 -b 1 -j 1 --dense_sampling \
--pretrained ./zoo/K400-I3D-ResNet-50-f32.pth --num_clips 1 --num_crops 1 --disable_scaleup