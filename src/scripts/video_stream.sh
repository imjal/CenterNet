python main.py ctdet  --exp_id bdd_video --batch_size 1 --lr 5e-4 --gpus 0,1,2,3 --num_workers 0 --num_epochs 10 \
    --arch dla_34  --num_iters 1210 \
    --dataset bddstream --vid_paths /data2/jl5/bdd100k/videos/100k/train/02133057-fc70cc0e.mov \
    --ann_paths /data2/jl5/bdd100k_pred/train/instances_02133057-fc70cc0e.pkl \
    --debug 4 --debugger_theme black --val_intervals 0 # --not_rand_crop --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth
    
    