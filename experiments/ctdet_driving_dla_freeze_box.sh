cd src
python main.py ctdet --exp_id driving_dla_freeze_box_epoch --freeze --load_model ../models/ctdet_coco_dla_2x.pth  --batch_size 32 --master_batch 5 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 8 --num_epochs 230 --lr_step 180,210 --dataset fifth
# python test.py ctdet --exp_id driving_dla_2x --keep_res --resume