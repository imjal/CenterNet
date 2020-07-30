#!/bin/bash
python online_distillation.py ctdet_stream  --exp_id acc_dla34_rdm_driving$1_32 --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/driving1/driving$1.mp4 \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_driving$1.pkl \
    --input_h 736 --input_w 1280 --input_res 1280 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 1 --save_video --debugger_theme white --not_rand_crop \
    --adaptive --display_timing --center_thresh 0.6 --vis_thresh 0.6 --lr 0.00001 \
    --momentum 0.9 --acc_collect --vidstream cv2 --umax 32