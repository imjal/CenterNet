cd src
python main.py ctdet --exp_id driving_resdcn18_fourth_vehicle_tmp --class_remap --arch resdcn_18 --load_model ../models/ctdet_coco_resdcn18.pth --batch_size 32 --master_batch 5 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 8 --num_epochs 100 --lr_step 180,210 --dataset half
# python test.py ctdet --exp_id driving_dla_2x --keep_res --resume