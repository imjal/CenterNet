python main.py ctdet  --exp_id bdd_dla_2x_overfit --batch_size 8 --lr 5e-4 --gpus 0,1,2,3 --num_workers 4 --num_epochs 25 \
    --arch dla_34 \
    --dataset bdd --input_h 736 --input_w 1280 --input_res 1280 \
    --test --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x_overfit/model_last.pth
    --debug 4 --debugger_theme black --val_intervals 0  --not_rand_crop
    
    