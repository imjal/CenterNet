python main.py ctdet --exp_id bdd_dla_2x_res --batch_size 8 --lr 5e-4 --gpus 0 --num_workers 0 --num_epochs 25 \
    --arch dla_34 --input_h 736 --input_w 1280 --input_res 1280 \
    --dataset bdd # --test --debug 4 --load_model /home/jl5/CenterNet/exp/ctdet/bdd_dla_2x/model_last.pth