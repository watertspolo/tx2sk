
name='sample_models_coco256'
CUDA_VISIBLE_DEVICES="0" python test_worker.py \
                                --dataset coco \
                                --model_name image_human256v2 \
                                --load_from_epoch 70 \
                                --test_sample_num 8 \
                                --batch_size 8 \
