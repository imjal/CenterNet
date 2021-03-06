python main.py ctdet_semseg  --exp_id resdcn_18_pt_seg_mAP80_thresh6_RMSProp_$1 --batch_size 1 --lr 5e-4 --gpus 0 --num_workers 1 --num_epochs 1 \
    --arch resdcn_18 \
    --num_iters 1210 \
    --dataset bddstream --vid_paths /data2/jl5/bdd100k/videos/100k/train/02133057-fc70cc0e.mov \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_02133057-fc70cc0e.pkl \
    --input_h 736 --input_w 1280 --input_res 1280 \
    --not_rand_crop --shift 0 --scale 0 --flip 0 --no_color_aug \
    --debug 4 --debugger_theme black --not_rand_crop \
    --display_timing --adaptive --center_thresh 0.6 --vis_thresh 0.6 --lr $1 --momentum 0.9 --acc_collect \
    --load_model /home/jl5/CenterNet/models/ctdet_coco_resdcn18.pth 
    # --not_rand_crop --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth