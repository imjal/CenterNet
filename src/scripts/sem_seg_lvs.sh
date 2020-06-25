python main.py ctdet_semseg  --exp_id semseg_lvs_tmp --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 \
    --num_iters 17824 \
    --dataset bddstream --vid_paths /data2/jl5/driving1/driving1000.mp4 \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_driving1000.pkl \
    --input_h 736 --input_w 1280 --input_res 3840 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_dla_2x.pth \
    --debug 4 --debugger_theme black --not_rand_crop --center_thresh 0.1
    # --replay_samples 4
    # --not_rand_crop --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth
    # 2160 x 3840