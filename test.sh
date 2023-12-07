python finetune.py \
    --data_root ./cus_dataset \
    --model_type vit_h \
    --checkpoint_path ./checkpoints/last.ckpt \
    --freeze_image_encoder \
    --freeze_mask_decoder \
    --freeze_prompt_encoder \
    --train_VPT_decoder \
    --batch_size 4 \
    --image_size 1024 \
    --steps 1500000 \
    --learning_rate 1.e-5 \
    --weight_decay 0.01 \
    --metrics_interval 37 \
    --test_only