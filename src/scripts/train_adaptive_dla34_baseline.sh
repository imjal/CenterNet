#!/bin/bash
python online_distillation.py ctdet_stream  --exp_id baseline_dla34_pt_$1 --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/bdd100k/videos/100k/train/02133057-fc70cc0e.mov \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_02133057-fc70cc0e.pkl \
    --input_h 736 --input_w 1280 --input_res 1280 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debugger_theme white --save_video --debug 1 \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_dla_2x.pth --umax 32 \
    --display_timing --center_thresh 0.5 --vis_thresh 0.5 --lr 0.0001 --momentum 0.9 --acc_collect --vidstream cv2