
name='sample_models_coco256'
CUDA_VISIBLE_DEVICES="0" python test_worker.py \
                                --dataset coco \
                                --model_name sample_models_coco256 \
                                --load_from_epoch 200 \
                                --test_sample_num 8 \
                                --batch_size 8 \
