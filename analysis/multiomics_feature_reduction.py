import pandas as pd
import numpy as np
import warnings
import wandb
import os
import torch
import matplotlib.pyplot as plt
from tabpfnwide.utils import PredictionResults
from tabpfnwide.data import get_wide_validation_datasets
from tabpfn.model_loading import load_model_criterion_config
import warnings
warnings.filterwarnings("ignore")
import argparse



def main(dataset_name, checkpoint_paths, output_file, device="cuda:0", omics_list=None):
    """
    Runs feature reduction experiments on multi-omics datasets using a specified model and checkpoints.
    Parameters:
        dataset_name (str): Name of the dataset to evaluate.
        checkpoint_paths (list of str): List of paths to model checkpoints. Use "default" for the base model.
        output_file (str): Path to the CSV file where results will be saved.
        device (str, optional): Device to run the model on (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".
        omics_list (list of str, optional): List of omics types to include. If None, defaults to "mRNA".
    For each checkpoint and a range of feature counts, the function:
    - Loads the model and checkpoint.
    - Iterates over validation splits of the dataset with reduced features.
    - Evaluates the model, collects predictions, and computes accuracy and weighted F1 score.
    - Appends results to a CSV file, skipping experiments that have already been run.
    """
    
    api = wandb.Api()
    results = pd.DataFrame(columns=[
        'Dataset', 'Omic', 'Checkpoint', 'n_features', 'Fold', 'Model',
        'Max_finetune', 'Accuracy', 'f1_weighted', 'prediction_probas', 'ground_truth'
    ])
    if os.path.exists(output_file):
        results = pd.read_csv(output_file)
    for checkpoint_path in checkpoint_paths:
        model, _, _ = load_model_criterion_config(
            model_path=None,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=False,
            which='classifier',
            version='v2.5',
            download=True,
        )
        if checkpoint_path != "default":
            model.features_per_group = 1
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint)
            name = checkpoint_path.split("/")[-1]
        else:
            name = "default"
            

        for n_features in [200, 500, 2000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 0]:
            exists = ((results['n_features'] == n_features) & (results['Checkpoint'] == checkpoint_path.split("/")[-1]) & (results['Dataset'] == dataset_name)).any()
            if exists:
                print(f"Skipping {dataset_name} with {n_features} features, already exists")
                continue
            print(f"Validating {dataset_name} with {n_features} features")
            model.to(device)
            model.eval()
            for i, dataset in enumerate(get_wide_validation_datasets(device, dataset_name=dataset_name, n_splits=5, n_repeats=1, reduced_features=n_features, omics=omics_list)):
                X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataset
                exists = ((results['n_features'] == X_train_tensor.shape[-1]) & (results['Checkpoint'] == checkpoint_path.split("/")[-1]) & (results['Dataset'] == dataset_name)).any()
                if exists and i == 0:
                    print(f"Skipping {dataset_name} with {X_train_tensor.shape[-1]} features, already exists")
                    break
                MemoryUsageEstimator.reset_peak_memory_if_required(
                    save_peak_mem=True,
                    model=model,
                    X=torch.cat([X_train_tensor, X_test_tensor], dim=0),
                    cache_kv=False,
                    dtype_byte_size= 2,
                    device=torch.device(device),
                    safety_factor=4, 
                )

                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_logits = model(
                            train_x=X_train_tensor,
                            train_y=y_train_tensor,
                            test_x=X_test_tensor,
                        )
                        n_classes = len(np.unique(y_train_tensor.cpu()))
                        pred_logits = pred_logits[..., :n_classes].float()  
                        pred_probs = torch.softmax(pred_logits, dim=-1)[:, 0, :].detach().cpu().numpy()
                pred_res = PredictionResults(y_test_tensor.flatten().cpu().numpy(), pred_probs)
                accuracy = pred_res.get_classification_report(print_report=False)['accuracy']
                f1_weighted = pred_res.get_f1_score(average='weighted')
                results = pd.concat([
                    results,
                    pd.DataFrame({
                        'Dataset': dataset_name,
                        'Omic': "+".join(omics_list) if omics_list else "mRNA",
                        'Checkpoint': checkpoint_path.split("/")[-1],
                        'n_features': X_train_tensor.shape[-1],
                        'Fold': i,
                        'Model': name,
                        'Accuracy': accuracy,
                        'f1_weighted': f1_weighted,
                        'prediction_probas': [" ".join(map(str, pred_res.prediction_probas))],
                        'ground_truth': [" ".join(map(str, pred_res.ground_truth))],
                    })
                ], ignore_index=True)
            model.to("cpu")
            results.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("checkpoints_dir",  type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("omics_list", type=str, nargs='+')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    checkpoints_dir = args.checkpoints_dir
    output_file = args.output_file
    checkpoint_paths = [os.path.join(checkpoints_dir, cp) for cp in os.listdir(checkpoints_dir) if cp.endswith('.pt')]
    checkpoint_paths += ["default"]
    omics_list = args.omics_list
    device = args.device
    main(dataset_name, checkpoint_paths, output_file, device, omics_list)
