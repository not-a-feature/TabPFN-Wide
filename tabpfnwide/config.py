from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_steps: int = 10000
    use_wandb: bool = False
    d_type: str = "float16"  # Options: "float16", "float32"
    warmup_proportion: int = 0.02
    num_cycles: int = 10
    n_estimators: int = 1
    gradient_clipping: float = 1.0
    validation_interval: int = 200
    validation_interval_wide: int = 200
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    resume_checkpoint: str = None
    feature_order: str = None
    use_original_model: bool = False
    model_path: str = None
    grouping: int | None = None
    validation_datasets = ["COAD", "gbm"]
    omic_combinations = [["mrna"], ["methylation"], ["mrna", "methylation", "mirna"]]


@dataclass
class FeatureAddingConfig:
    add_features_min: int = 0
    add_features_max: int = 0
    warmup_steps: int = 0
    min_sparsity: float = 0.0
    max_sparsity: float = 0.2
    min_noise: float = 0.0
    max_noise: float = 0.1
    use_mlp: bool = False
    include_original: bool = True


@dataclass
class PriorDatasetConfig:
    batch_size: int = 8
    batch_size_per_gp: int = 8  # Number of datasets per group, sharing similar characteristics
    device: str = "cpu"
    min_features: int = 10
    max_features: int = 100
    max_classes: int = 10
    min_seq_len: int = 40
    max_seq_len: int = 400
    log_seq_len: bool = False
    seq_len_per_gp: bool = False
    min_train_size: float = 0.3
    max_train_size: float = 0.9
    replay_small: bool = False
    prior_type: str = "mlp_scm"  # default
    n_jobs: int = 1  # Set to 1 to avoid nested parallelism


@dataclass
class PriorDataLoaderConfig:
    batch_size: int = None
    shuffle: bool = False
    num_workers: int = 1
    prefetch_factor: int = 4
    pin_memory: bool = True
    pin_memory_device: str = "cuda"  # Device to pin memory to, typically the training device
