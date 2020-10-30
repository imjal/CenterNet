#!/bin/bash
python online_distillation.py ctdet_stream --exp_id resdcn18_adaptive_driving_ap75_ $1 \
    --batch_size 1 --lr 5e-$3 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch resdcn_18 --adaptive \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/driving1/driving$1.mp4 \
    --ann_paths /data2/jl5/mmdetect_results/driving$1/coco_images.json \
    --input_h 736 --input_w 1280 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 1 --debugger_theme white --not_rand_crop \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_resdcn18.pth \
    --adaptive --display_timing --center_thresh 0.3 --vis_thresh 0.3 \
    --momentum 0.9 --acc_collect --vidstream cv2 --umax $2  --acc_thresh 100 # --tracking
