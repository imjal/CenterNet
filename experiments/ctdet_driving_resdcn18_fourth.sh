cd src
python main.py ctdet --exp_id driving_resdcn18_fourth_coco --arch resdcn_18 --load_model /data2/jl5/centdt_exp/ctdet/driving_resdcn18_fourth_2/model_best.pth  --batch_size 32 --master_batch 5 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 8 --num_epochs 230 --lr_step 180,210 --dataset fourth
# python test.py ctdet --exp_id driving_dla_2x --keep_res --resume