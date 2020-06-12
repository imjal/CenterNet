python main.py ctdet  --exp_id bdd_video_batch_og_res --batch_size 4 --lr 5e-4 --gpus 0,1,2,3 --num_workers 2 --num_epochs 25 \
    --arch dla_34  --num_iters 100 \
    --dataset bddstream --vid_paths /data2/jl5/bdd100k/videos/100k/train/02133057-fc70cc0e.mov --input_h 736 --input_w 1280 --input_res 1280 \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_02133057-fc70cc0e.pkl \
    --debug 4 --debugger_theme black --val_intervals 0 # --not_rand_crop --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth
    
    