import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np
import torch
import random
import wandb
import pandas as pd
from typing import Dict, List, Tuple
from data_utils import set_seeds
from hyperparam_trainer_gcn import HyperParamTrainer
from hyperparam_trainer_gat import HyperParamTrainerGAT
from sklearn.model_selection import KFold


K_FOLDS = 3

# Main function to set up and run the hyperparameter sweep
def main(project_name=None):
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'model': {
                'values': ['gcn']  # or 'gat' Model type to tune, ran individually
            },
            'in_features': {'value': 2},  # Fixed parameter
            'out_features': {'value': 1},  # Fixed parameter
            'embedding_dim': {
                # 32, 64, 128,
                'values': [16, 32, 64, 128, 256]
            },
            'hidden_dim': {
                'values': [32, 64, 128, 256, 512]
            },
            'num_layers': {
                # 2, 3, 4,
                'values': [2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'dropout': {
                'min': 0.0,
                'max': 0.5,
                'distribution': 'uniform'
            },
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01,
                'distribution': 'uniform'
            },
            'lr_patience': {
                'value': 5  # Learning rate scheduler patience
            },
            'pos_weight': {
                'values': [1.25, 1.5, 1.75, 2, 2.25]
            },
            'threshold': {
                'min': 0.0,
                'max': 1.0,
                'distribution': 'uniform'
            },
            'batch_size': {
                # 16, 32, 64, 
                'values': [16, 32, 64, 128]
            },
            'epochs': {
                'value': 250  # Maximum number of epochs
            },
            'patience': {
                'values': [10, 30, 50, 70, 90]  # Early stopping patience
            },
            
        }
    }

    set_seeds(42)
    # Load dataset
    graphs_flat, _ = dgl.load_graphs('../train_all.bin')
    # graphs_non_flat, _ = dgl.load_graphs('../non_flat_hetero_graph.bin')
    all_graphs = graphs_flat # + graphs_non_flat
    n_samples = len(all_graphs)

    # Shuffle data once before splitting
    random.shuffle(all_graphs)
    
    # Set up k-fold cross validation
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(kf.split(range(n_samples)))
    
    # Create the sweep
    # sweep_id = wandb.sweep(sweep_config, project=project_name)

    # Define the training function that will be called by wandb
    def train_sweep():
        print("NEW SWEEP")
        run = wandb.init(config=wandb.config) 

        # print(f"Current run hyperparameters:")
        # print(f"Batch size: {wandb.config.batch_size}")
        # print(f"Learning rate: {wandb.config.learning_rate}")
        # print(f"Hidden dim: {wandb.config.hidden_dim}")

        try:
            print("Starting fold processing")
            print(f"Number of fold indices: {len(list(fold_indices))}")
        
            f1_scores = []

            # For each fold
            for fold, (train_idx, val_idx) in enumerate(fold_indices):
                try:
                    print(f"Training fold {fold+1}/{K_FOLDS}")
                    
                    # Create train and validation datasets for this fold
                    train_dataset = [all_graphs[i] for i in train_idx]
                    val_dataset = [all_graphs[i] for i in val_idx]
                    
                    print(f"Fold {fold+1}: train: {len(train_dataset)} val: {len(val_dataset)}")
                
                    # Create dataloaders
                    batch_size = wandb.config.batch_size
                    train_loader = GraphDataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False
                    )
                    val_loader = GraphDataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False
                    )
                
                    # Initialize trainer
                    trainer = HyperParamTrainer(run, config=wandb.config)
                    
                    # Train and evaluate
                    f1, accuracy, recall, precision = trainer.train(train_loader, val_loader, fold)

                    # Store metrics
                    f1_scores.append(f1)
                    run.log({
                        f"fold-{fold} best f1": f1,
                        f"fold-{fold} best accuracy": accuracy,
                        f"fold-{fold} best precision": precision,
                        f"fold-{fold} best recall": recall,
                    })
                except torch.OutOfMemoryError:
                    print(f"OOM error in fold {fold+1}")
                    torch.cuda.empty_cache()
                    run.log({
                        f"fold-{fold} f1": -1,
                        f"fold-{fold} accuracy": -1,
                        f"fold-{fold} precision": -1,
                        f"fold-{fold} recall": -1,
                    })
            
            # Average performance across all folds for this config
            if len(f1_scores) == K_FOLDS:
                avg_val_f1 = sum(f1_scores) / K_FOLDS

                # Log the average - this is what Bayesian optimization will use
                run.log({"success": True, "val_f1": avg_val_f1})
            else:
                avg_val_f1 = 0
                run.log({"success": False, "val_f1": avg_val_f1})
        
        except Exception as e:
            print(f"Error in sweep: {str(e)}")
            run.summary.update({"success": False, "val_f1": 0})
            run.log({"error": str(e)})

    
    # Run the sweep
    if sweep_config.get('model', {}).get('values', [None])[0] == 'gcn':
        wandb.agent("khf0blnu", train_sweep, project="gcn-learnable-tuning", count=2)
    elif sweep_config.get('model', {}).get('values', [None])[0] == 'gat':
        wandb.agent("grrmlu8b", train_sweep, project="gat-learnable-tuning", count=1)
    return sweep_id

# Function to extract and analyze the best parameters after sweep
def analyze_sweep_results(sweep_id, project_name=None):
    """
    Analyze the results of a hyperparameter sweep and extract the best configuration.
    
    Args:
        sweep_id: ID of the sweep to analyze
        project_name: Name of the W&B project
    
    Returns:
        best_config: Dictionary containing the best hyperparameter configuration
    """
    # Initialize W&B API
    api = wandb.Api()
    
    # Get the sweep
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    
    # Get all runs in the sweep
    runs = sweep.runs
    
    # Extract metrics and config for each completed run
    results = []
    for run in runs:
        if run.state == "finished":
            # Get the run summary (metrics)
            summary = {k: v for k, v in run.summary.items() 
                      if not k.startswith('_')}
            
            if summary.get('success', False):
                # Get the run config (hyperparameters)
                config = {k: v for k, v in run.config.items() 
                         if not k.startswith('_')}
                
                # Combine metrics and config
                result = {**summary, **config}
                results.append(result)
    
    # Create a DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Sort by validation F1 score (our optimization metric)
        results_df = results_df.sort_values('val_f1', ascending=False)
        
        # Get the best configuration
        best_config = results_df.iloc[0].to_dict()
    else:
        best_config = {}
    
    return best_config

if __name__ == "__main__":
    wandb.login(key="d4d35374d1ffcaec0b132f70e473a21570bf34ee")
    project_name = "gcn-learnable-tuning" ## or "gat-learnable-tuning"


    # sweep_id = main(project_name)
    # print(sweep_id)
    sweep_id = "khf0blnu"
    best_config = analyze_sweep_results(sweep_id, project_name)

    # Write the best configuration to file
    output_file = "gcn_learnable_params" ## or "gat_learnable_params"
    with open(output_file, 'w') as f:
        if best_config:
            f.write("Best Hyperparameter Configuration:\n")
            f.write(f"Validation F1 Score: {best_config.get('val_f1', 'N/A')}\n")
            f.write("\nHyperparameters:\n")
            for param in ['embedding_dim', 'hidden_dim', 'num_layers', 'dropout', 'learning_rate', 
                        'batch_size', 'pos_weight', 'threshold', 'patience']:
                f.write(f"- {param}: {best_config.get(param, 'N/A')}\n")
    
    print(f"Best configuration saved to {output_file}")
    wandb.finish()