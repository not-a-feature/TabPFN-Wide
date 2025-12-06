import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    ### Train config
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=10000, help="Total number of training steps."
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weight and Biases for logging."
    )
    parser.add_argument(
        "--d_type",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for training.",
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.02,
        help="Proportion of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=10,
        help="Number of cycles for the cosine learning rate scheduler.",
    )
    parser.add_argument(
        "--gradient_clipping", type=float, default=1.0, help="Gradient clipping value."
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=200,
        help="Interval (in steps) to perform validation.",
    )
    parser.add_argument(
        "--validation_interval_wide",
        type=int,
        default=200,
        help="Interval (in steps) to perform wide validation.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Interval (in steps) to save model checkpoints.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--feature_order",
        type=str,
        default=None,
        help="Order of features to be used during training for causal models. If None, uses the order from the dataset.",
    )
    parser.add_argument(
        "--use_original_model", action="store_true", help="Use the original pretrained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the original model to be used for finetuning. If None, uses the default model path.",
    )

    ### Feature adding config
    parser.add_argument(
        "--add_features_min",
        type=int,
        default=0,
        help="Minimum number of features to add during training using sparse linear projection.",
    )
    parser.add_argument(
        "--add_features_max",
        type=int,
        default=0,
        help="Maximum number of features to add during training using sparse linear projection.",
    )
    parser.add_argument(
        "--feature_adding_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for feature adding. If 0, no warmup is applied.",
    )
    parser.add_argument(
        "--max_sparsity_feature_adding",
        type=float,
        default=0.01,
        help="Sparsity for feature adding during training.",
    )
    parser.add_argument(
        "--max_noise_feature_adding",
        type=float,
        default=0.1,
        help="Noise level for feature adding during training.",
    )
    parser.add_argument(
        "--feature_adding_use_mlp",
        action="store_true",
        help="Use MLP for feature adding instead of linear projection.",
    )
    parser.add_argument(
        "--feature_adding_dismiss_original",
        action="store_true",
        help="Dismiss original features when adding new features during training.",
    )

    ### Model config
    parser.add_argument(
        "--model_emsize", type=int, default=192, help="Embedding size for the model."
    )
    parser.add_argument(
        "--model_features_per_group", type=int, default=1, help="Number of features per group."
    )
    parser.add_argument(
        "--model_max_num_classes",
        type=int,
        default=10,
        help="Maximum number of classes for classification tasks.",
    )
    parser.add_argument(
        "--model_nlayers", type=int, default=24, help="Number of layers in the model."
    )
    parser.add_argument(
        "--model_nhead", type=int, default=3, help="Number of attention heads in the model."
    )
    parser.add_argument(
        "--model_nhid_factor",
        type=int,
        default=2,
        help="Hidden dimension factor for the MLP layers.",
    )
    parser.add_argument(
        "--model_remove_duplicate_features",
        action="store_true",
        help="Remove duplicate features from the dataset.",
    )
    parser.add_argument(
        "--model_num_buckets",
        type=int,
        default=5000,
        help="Number of buckets for feature bucketing.",
    )
    parser.add_argument(
        "--model_recompute_layer",
        action="store_true",
        help="Enable activation checkpointing for each layer.",
    )
    parser.add_argument(
        "--model_recompute_attn",
        action="store_true",
        help="Enable activation checkpointing for attention layers.",
    )
    parser.add_argument(
        "--model_multiquery_item_attention",
        action="store_true",
        help="Use multiquery for attention between items.",
    )
    parser.add_argument(
        "--model_max_num_features",
        type=int,
        default=85,
        help="Maximum number of features in the dataset.",
    )
    parser.add_argument(
        "--model_feature_attention_type",
        type=str,
        default="full",
        choices=["full", "linear", "mamba"],
        help="Type of attention to use for features.",
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
        help="Seed for the model.",
    )
    parser.add_argument(
        "--model_num_thinking_rows",
        type=int,
        default=0,
        help="Number of thinking rows to prepend to each dataset.",
    )

    ### Prior dataset config
    parser.add_argument(
        "--prior_batch_size", type=int, default=8, help="Batch size for the prior dataset."
    )
    parser.add_argument(
        "--prior_batch_size_per_gp",
        type=int,
        default=8,
        help="Number of datasets per group in the prior dataset.",
    )
    parser.add_argument(
        "--prior_device_prior", type=str, default="cpu", help="Device for the prior dataset."
    )
    parser.add_argument(
        "--prior_min_features",
        type=int,
        default=10,
        help="Minimum number of features in the prior dataset.",
    )
    parser.add_argument(
        "--prior_max_features",
        type=int,
        default=100,
        help="Maximum number of features in the prior dataset.",
    )
    parser.add_argument(
        "--prior_max_classes",
        type=int,
        default=10,
        help="Maximum number of classes in the prior dataset.",
    )
    parser.add_argument(
        "--prior_min_seq_len",
        type=int,
        default=40,
        help="Minimum sequence length in the prior dataset.",
    )
    parser.add_argument(
        "--prior_max_seq_len",
        type=int,
        default=400,
        help="Maximum sequence length in the prior dataset.",
    )
    parser.add_argument(
        "--prior_log_seq_len",
        action="store_true",
        help="Use logarithmic sampling for sequence lengths in the prior dataset.",
    )
    parser.add_argument(
        "--prior_seq_len_per_gp",
        action="store_true",
        help="Sample sequence length per group in the prior dataset.",
    )
    parser.add_argument(
        "--prior_min_train_size",
        type=float,
        default=0.3,
        help="Minimum training size as a fraction of the total dataset size in the prior dataset.",
    )
    parser.add_argument(
        "--prior_max_train_size",
        type=float,
        default=0.9,
        help="Maximum training size as a fraction of the total dataset size in the prior dataset.",
    )
    parser.add_argument(
        "--prior_replay_small",
        action="store_true",
        help="Replay small datasets in the prior dataset.",
    )
    parser.add_argument(
        "--prior_type", type=str, default="mlp_scm", help="Type of prior dataset to use."
    )
    parser.add_argument(
        "--prior_n_jobs",
        type=int,
        default=1,
        help="Number of jobs for loading the prior dataset. Set to 1 to avoid nested parallelism.",
    )

    ### Prior dataloader config
    parser.add_argument(
        "--prior_dataloader_num_workers",
        type=int,
        default=1,
        help="Number of workers for the prior dataloader.",
    )
    parser.add_argument(
        "--prior_dataloader_prefetch_factor",
        type=int,
        default=4,
        help="Prefetch factor for the prior dataloader.",
    )
    parser.add_argument(
        "--prior_dataloader_pin_memory",
        action="store_true",
        help="Pin memory for the prior dataloader.",
    )

    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary for easier access
    parsed_args = {
        "train_config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_steps": args.num_steps,
            "use_wandb": args.use_wandb,
            "d_type": args.d_type,
            "warmup_proportion": args.warmup_proportion,
            "num_cycles": args.num_cycles,
            "gradient_clipping": args.gradient_clipping,
            "validation_interval": args.validation_interval,
            "validation_interval_wide": args.validation_interval_wide,
            "checkpoint_dir": args.checkpoint_dir,
            "save_interval": args.save_interval,
            "resume_checkpoint": args.resume_checkpoint,
            "feature_order": args.feature_order,
            "use_original_model": args.use_original_model,
            "model_path": args.model_path,
        },
        "feature_adding_config": {
            "add_features_min": args.add_features_min,
            "add_features_max": args.add_features_max,
            "warmup_steps": args.feature_adding_warmup_steps,
            "max_sparsity": args.max_sparsity_feature_adding,
            "max_noise": args.max_noise_feature_adding,
            "use_mlp": args.feature_adding_use_mlp,
            "include_original": not args.feature_adding_dismiss_original,
        },
        "model_config": {
            "emsize": args.model_emsize,
            "features_per_group": args.model_features_per_group,
            "max_num_classes": args.model_max_num_classes,
            "nlayers": args.model_nlayers,
            "nhead": args.model_nhead,
            "nhid_factor": args.model_nhid_factor,
            "remove_duplicate_features": args.model_remove_duplicate_features,
            "num_buckets": args.model_num_buckets,
            "max_num_features": args.model_max_num_features,
            "feature_attention_type": args.model_feature_attention_type,
            "recompute_layer": args.model_recompute_layer,
            "recompute_attn": args.model_recompute_attn,
            "multiquery_item_attention": args.model_multiquery_item_attention,
            "seed": args.model_seed,
            "num_thinking_rows": args.model_num_thinking_rows,
        },
        "prior_dataset_config": {
            "batch_size_per_gp": args.prior_batch_size_per_gp,
            "device": args.prior_device_prior,
            "min_features": args.prior_min_features,
            "max_features": args.prior_max_features,
            "max_classes": args.prior_max_classes,
            "min_seq_len": args.prior_min_seq_len,
            "max_seq_len": args.prior_max_seq_len,
            "log_seq_len": args.prior_log_seq_len,
            "seq_len_per_gp": args.prior_seq_len_per_gp,
            "min_train_size": args.prior_min_train_size,
            "max_train_size": args.prior_max_train_size,
            "replay_small": args.prior_replay_small,
            "prior_type": args.prior_type,
            "n_jobs": args.prior_n_jobs,
        },
        "prior_dataloader_config": {
            "num_workers": args.prior_dataloader_num_workers,
            "prefetch_factor": args.prior_dataloader_prefetch_factor,
            "pin_memory": args.prior_dataloader_pin_memory,
        },
    }

    return parsed_args
