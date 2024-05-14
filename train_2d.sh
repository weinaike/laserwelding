python train_2d.py \
-b 8 \
-j 16 \
-lr 1e-2 \
--step 8 \
--epochs 10 \
--model_file snapshots/merge_cls_resnet18.pth \
--gpu 1 \
--log_file "snapshots/merge_cls_resnet18.log" 

