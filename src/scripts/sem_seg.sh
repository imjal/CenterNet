python main.py ctdet_semseg  --exp_id adaptive_pretrained_no_aug_tmp --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch dla_34 \
    --num_iters 1210 \
    --dataset bddstream --vid_paths /data2/jl5/bdd100k/videos/100k/train/02133057-fc70cc0e.mov \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_02133057-fc70cc0e.pkl \
    --input_h 736 --input_w 1280 --input_res 1280 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 4 --debugger_theme black --not_rand_crop \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_dla_2x.pth
    # --not_rand_crop --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth