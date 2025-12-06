torchrun --master_port 29501  $BASE_DIR_LOCAL/training/train.py \
            --batch_size 16 \
            --learning_rate 1e-5 \
            --weight_decay 1e-4 \
            --num_steps 10 \
            --d_type float16 \
            --warmup_proportion 0.02 \
            --num_cycles 10 \
            --gradient_clipping 1.0 \
            --validation_interval 200 \
            --validation_interval_wide 100 \
            --add_features_min 200 \
            --add_features_max 1500 \
	        --max_sparsity_feature_adding 0.05 \
	        --max_noise_feature_adding 1.0 \
            --use_original_model \
\
            --model_emsize 192 \
            --model_features_per_group 1 \
            --model_max_num_classes 10 \
            --model_nlayers 24 \
            --model_nhead 3 \
            --model_nhid_factor 2 \
            --model_num_buckets 5000 \
            --model_max_num_features 85 \
            --model_feature_attention_type full \
            --model_seed 42 \
            --model_num_thinking_rows 64 \
\
            --prior_batch_size_per_gp 4 \
            --prior_device_prior cpu \
            --prior_min_features 50 \
            --prior_max_features 350 \
            --prior_max_classes 10 \
            --prior_min_seq_len 40 \
            --prior_max_seq_len 300 \
            --prior_log_seq_len \
            --prior_min_train_size 0.3 \
            --prior_max_train_size 0.9 \
            --prior_type mlp_scm \
            --prior_n_jobs 1 \
\
            --prior_dataloader_num_workers 1 \
            --prior_dataloader_prefetch_factor 4 \
            --prior_dataloader_pin_memory \
\
            --checkpoint_dir $BASE_DIR_LOCAL/checkpoints/ \
            --save_interval 10 \
