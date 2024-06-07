python3 test_3d.py --temporal_module_name TAM  --backbone_net resnet  --depth 18  --without_t_stride \
    -e --datadir images/ --dataset laser_welding  --modality rgb --augmentor_ver v3 --frames_per_group 1 --groups 8 --threed_data \
    --logdir snapshots/ -b 1 -j 1  --num_clips 1 --num_crops 1 --disable_scaleup  \
    --pretrained snapshots/laser_welding-rgb-TAM-b3-sum-resnet-18-f8-cosine-bs16-e50/model_best.pth.tar \
    --debug

