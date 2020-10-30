#!/bin/bash
python online_distillation.py ctdet_stream  --exp_id results_dla34_baseline_pt_oxford_$1 --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/OxfordCarVideos/$1-centre-vid.mp4 \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_driving1000.pkl \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 1 --debugger_theme white --not_rand_crop \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_dla_2x.pth \
    --display_timing --center_thresh 0.3 --vis_thresh 0.3 --lr 0.00001 \
    --momentum 0.9 --acc_collect --vidstream cv2 --umax 32
    # --not_rand_crop --test --load_mbuodel /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth
