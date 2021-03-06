#!/bin/bash
python online_distillation.py ctdet_stream  --exp_id driving_resdcn18_half_vehicle_model25_$1 --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch resdcn_18 \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/driving1/driving$1.mp4 \
    --ann_paths /data2/jl5/mmdetect_results/driving$1/coco_images.json \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 1 --debugger_theme white --not_rand_crop \
    --load_model /data2/jl5/centdt_exp/ctdet/driving_resdcn18_half_vehicle/model_18.pth \
    --center_thresh 0.3 --vis_thresh 0.3 --lr 0.00001 \
    --momentum 0.9 --acc_collect --vidstream cv2 --umax 32 --save_video
    # --not_rand_crop --test --load_mbuodel /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth