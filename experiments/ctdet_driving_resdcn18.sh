cd src
python main.py ctdet --exp_id driving_resdcn18_epoch_freeze_wh --arch resdcn_18 --freeze --load_model ../models/ctdet_coco_resdcn18.pth  --batch_size 32 --master_batch 5 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 8 --num_epochs 230 --lr_step 180,210 --dataset fifth
# python test.py ctdet --exp_id driving_dla_2x --keep_res --resume