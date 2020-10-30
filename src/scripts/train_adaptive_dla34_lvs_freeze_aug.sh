#!/bin/bash
python online_distillation.py ctdet_stream --exp_id results_mmdet_dla34_pt_freeze_driving$1_$2_$3 --batch_size 1 \
    --lr 5e-$3 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 --adaptive \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/driving1/driving$1.mp4 \
    --ann_paths /data2/jl5/mmdetect_results/driving$1/coco_images.json \
    --not_rand_crop --augment_stream \
    --debug 4 --debugger_theme white \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_dla_2x.pth \
    --display_timing --center_thresh 0.3 --vis_thresh 0.3 \
    --momentum 0.9 --acc_collect --vidstream cv2 --umax $2 # --tracking
