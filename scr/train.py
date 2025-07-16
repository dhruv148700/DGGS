import dgl 
import torch
import torch.nn as nn
import numpy as np 
import copy
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from dgl.dataloading import GraphDataLoader

from data_utils import set_seeds, get_confidence_interval, split_data
from plot_metrics import plot_metrics
from GCN_learnable import GCNLearnableModel
from GAT_learnable import GATLearnableModel
import argparse

USE_GPU = True

dtype = torch.float32 
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)


def get_model_params(model):
    return {name: param.clone() for name, param in model.named_parameters()}

def evaluate_model(dataloader, model, epoch, sub_folder, criterion = None, test_set = False, seed=42):
    model.eval()
    total_correct = 0  # To track total correct predictions
    total_examples = 0
    total_loss = 0.0

    # These lists will store all predictions and labels for metric computation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batched_graph in dataloader:

            batched_graph = batched_graph.to(device)

            inputs = {
                'assmpt': batched_graph.nodes['assmpt'].data['features'],
                'rule': batched_graph.nodes['rule'].data['features'],
                'non_assmpt': batched_graph.nodes['non_assmpt'].data['features']
            }

            labels = batched_graph.nodes['assmpt'].data['label']

            outputs = model(batched_graph, inputs)
            logits = outputs['assmpt'].squeeze(1)
            
            #calculate loss 
            if criterion:
                loss = criterion(logits, labels.float())
                total_loss += (loss * labels.size(0)).item()

            predictions = (torch.sigmoid(logits) > THRESHOLD).long()

            correct = (predictions == labels).sum().item()
            total_correct += correct

            total_examples += labels.size(0)

            # Save all preds/labels for precision, recall, f1
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = total_correct / total_examples
        avg_loss = total_loss / total_examples

        # Compute precision, recall, and F1
        # For binary classification, these default to 'binary' if your labels are in {0, 1}.
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if test_set:
            with open(f"../{sub_folder}/output.txt", "a") as file: 
                file.write(f"Test Set  F1 = {f1:.4f} Accuracy = {accuracy:.4f} Precision = {precision:.4f} Recall = {recall:.4f} \n")
        else: 
            with open(f"../{sub_folder}/output.txt", "a") as file:
                file.write(f"Epoch {epoch}: Val Set Loss = {avg_loss:.4f} F1 = {f1:.4f} Accuracy = {accuracy:.4f} Precision = {precision:.4f} Recall = {recall:.4f} \n")

    return (f1, accuracy, precision, recall) if test_set else avg_loss


def train_model(model, optimizer, scheduler, train_loader, val_loader, sub_folder, epochs=1, seed=42):
    pos_weight = torch.tensor([POS_WEIGHT]).to(device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    model = model.to(device=device)

    training_losses = []
    validation_losses = []
    
    #Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    best_epoch = 0

    for e in range(epochs):
        print(f"{e=}")
        model.train()
        train_loss = 0.0
        total_nodes = 0
        initial_params = get_model_params(model)

        for batched_graph in train_loader:
            batched_graph = batched_graph.to(device)

            inputs = {
                'assmpt': batched_graph.nodes['assmpt'].data['features'],
                'rule': batched_graph.nodes['rule'].data['features'],
                'non_assmpt': batched_graph.nodes['non_assmpt'].data['features']
            }

            labels = batched_graph.nodes['assmpt'].data['label']
        
            outputs = model(batched_graph, inputs)
            logits = outputs['assmpt'].squeeze(1)
            
            loss = loss_fn(logits, labels.float())

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters of the model using the gradients
            optimizer.step()

            # Weight the loss by number of nodes in this graph
            num_nodes = labels.size(0)
            train_loss += (loss * num_nodes).item()
            total_nodes += num_nodes

        avg_train_loss = train_loss / total_nodes
        training_losses.append(avg_train_loss)

        f_out = f"../{sub_folder}/output.txt"
        print(f_out)
        with open(f_out, "a") as file:
            file.write('Epoch: %d, training loss = %.4f\n' % (e, avg_train_loss))
        
        final_params = get_model_params(model)
        for name in initial_params:
            if (
                torch.equal(initial_params[name], final_params[name]) and 
                "conv2.mods.('rule', 'supports', 'rule')" not in name and 
                "conv2.mods.('non_assmpt', 'supports', 'non_assmpt')" not in name and 
                "conv2.mods.('rule', 'derives', 'non_assmpt')" not in name and 
                "conv2.mods.('assmpt', 'supports', 'rule')" not in name and 
                "conv2.mods.('non_assmpt', 'supports', 'rule')" not in name and 
                "conv1.mods.('assmpt', 'supports', 'rule')" not in name and
                "conv1.mods.('non_assmpt', 'supports', 'rule')" not in name and
                "conv1.mods.('assmpt', 'supports', 'rule')" not in name and
                "conv1.mods.('rule', 'supports', 'rule')" not in name
            ):
                print(f"Warning: Parameter {name} did not change during epoch {e}.")
        
        avg_val_loss = evaluate_model(val_loader, model, e, sub_folder, criterion=loss_fn, test_set=False, seed=seed)
        validation_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        with open(f_out, "a") as file:
            file.write('Epoch: %d, learning rate = %.6f, counter: %d, best: %.6f\n' % (e, current_lr, scheduler.num_bad_epochs, scheduler.best),)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
            best_epoch = e
        else:
            counter += 1
        
        
        # Check if early stopping criteria is met 
        if counter >= PATIENCE:
            with open(f_out, "a") as file:
                file.write(f"Early stopping triggered at epoch {e}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}\n")
            break
    
    # # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return training_losses, validation_losses

def calculate_bootstrap_confidence_intervals(dataset, model, sub_folder, n_bootstrap=10, sample_fraction=0.75, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for model performance metrics by calling evaluate_model
    on multiple bootstrap samples.
    
    Parameters:
    -----------
    dataset : torch dataset
        The test dataset
    model : torch model
        The trained model
    sub_folder : str
        Folder to save results
    n_bootstrap : int
        Number of bootstrap samples to use (default: 10)
    sample_fraction : float
        Fraction of dataset to use in each bootstrap sample (default: 0.75)
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    dict
        Dictionary containing the mean, standard deviation, and confidence intervals for each metric
    """
    # Convert dataset to a list for easier sampling
    dataset_list = list(dataset)
    n_samples = len(dataset_list)
    sample_size = int(n_samples * sample_fraction)
    
    # Lists to store bootstrap results
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    # Create a temporary subfolder for bootstrap samples
    # This avoids writing to the output file for each bootstrap iteration
    bootstrap_subfolder = f"{sub_folder}/bootstrap_temp"
    os.makedirs(f"../{bootstrap_subfolder}", exist_ok=True)
    
    # Run bootstrap iterations
    for b in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(range(n_samples), size=sample_size, replace=True)
        bootstrap_dataset = [dataset_list[i] for i in indices]

        test_loader = GraphDataLoader(
            bootstrap_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS
        )
        
        f1, accuracy, precision, recall = evaluate_model(
            test_loader, model, epoch=b, 
            sub_folder=bootstrap_subfolder,
            test_set=True  # This ensures we get the metrics tuple back
        )
        
        # Store results
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Calculate and format results
    f1_mean, f1_std, f1_lower, f1_upper = get_confidence_interval(f1_scores)
    acc_mean, acc_std, acc_lower, acc_upper = get_confidence_interval(accuracy_scores)
    prec_mean, prec_std, prec_lower, prec_upper = get_confidence_interval(precision_scores)
    rec_mean, rec_std, rec_lower, rec_upper = get_confidence_interval(recall_scores)
    
    # Add a summary to the main output file as well
    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write("\nBootstrap Confidence Intervals (95%):\n")
        file.write(f"Number of bootstrap samples: {n_bootstrap}, Sample size: {sample_size}/{n_samples}\n\n")
        file.write(f"F1: {f1_mean:.4f} ± {f1_std:.4f} [{f1_lower:.4f}, {f1_upper:.4f}]\n")
        file.write(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f} [{acc_lower:.4f}, {acc_upper:.4f}]\n")
        file.write(f"Precision: {prec_mean:.4f} ± {prec_std:.4f} [{prec_lower:.4f}, {prec_upper:.4f}]\n")
        file.write(f"Recall: {rec_mean:.4f} ± {rec_std:.4f} [{rec_lower:.4f}, {rec_upper:.4f}]\n")
    
    # Return the results as a dictionary
    return {
        'f1': (f1_mean, f1_std, f1_lower, f1_upper),
        'accuracy': (acc_mean, acc_std, acc_lower, acc_upper),
        'precision': (prec_mean, prec_std, prec_lower, prec_upper),
        'recall': (rec_mean, rec_std, rec_lower, rec_upper)
    }



def train_and_evaluate(model_type, epochs=50, sub_folder=None):
    """
    Run k-fold cross-validation with the given parameters.
    
    Args:
        k: Number of folds for cross-validation
        base_seed: Base seed value to derive individual fold seeds
        epochs: Number of training epochs per fold
        sub_folder: Output directory for results
    
    Returns:
        Dictionary containing metrics from all folds
    """

    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write(f"Starting training\n")

    # Initialize this model when using learnable embeddings 
    if model_type == 'gcn':
        model = GCNLearnableModel(
            in_features=2, 
            hidden_features=HIDDEN_DIM, 
            out_features=1, 
            embedding_dim=EMBEDDING_DIM, 
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT
        )
    elif model_type == 'gat':
        model = GATLearnableModel(
            in_features=2, 
            hidden_features=HIDDEN_DIM, 
            out_features=1, 
            embedding_dim=EMBEDDING_DIM, 
            num_layers=NUM_LAYERS, 
            dropout=DROPOUT
        )
    else:
        raise ValueError("model_type must be either 'gcn' or 'gat'")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # Reduce LR when monitored quantity stops decreasing
        factor=0.75,        # Multiply LR by this factor
        patience=5,        # Number of epochs with no improvement after which LR will be reduced
        verbose=True,      # Print message when LR is reduced
        min_lr=1e-6        # Lower bound on the learning rate
    ) 

    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total number of parameters is: {}".format(params))

    # Load and split data
    all_train_data, _ = dgl.load_graphs('../train_all.bin')
    test_set_25_100, _ = dgl.load_graphs('../test_25_100.bin')
    test_set_iccma, _ = dgl.load_graphs('../test_iccma.bin') 
    test_set_all, _ = dgl.load_graphs('../test_all.bin')

    train_set, val_set = split_data(all_train_data, None)

    # Create DataLoaders for batch training
    train_loader = GraphDataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    val_loader = GraphDataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )
    
    test_25_100_loader = GraphDataLoader(
        test_set_25_100,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    test_iccma_loader = GraphDataLoader(
        test_set_iccma,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    test_all_loader = GraphDataLoader(
        test_set_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS
    )


    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set 25/100: {len(test_set_25_100)}, All ICCMA Test set: {len(test_set_iccma)} All instances Test set: {len(test_set_all)}\n")

    # Train model and collect metrics
    train_losses, val_losses = train_model(model, optimizer, scheduler, train_loader, val_loader, sub_folder, epochs=epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), f'../{sub_folder}/trained_model.pt')

    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write(f"\nICCMA results - 25/100 element files\n")

    (f1, accuracy, precision, recall) = evaluate_model(test_25_100_loader, model, 0, sub_folder, test_set=True)
    bootstrap_25_100 = calculate_bootstrap_confidence_intervals(test_set_25_100, model, sub_folder)

    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write(f"\nICCMA results - all element files\n")
    
    (f1_rest, accuracy_rest, precision_rest, recall_rest) = evaluate_model(test_iccma_loader, model, 0, sub_folder, test_set=True)
    bootstrap_iccma = calculate_bootstrap_confidence_intervals(test_set_iccma, model, sub_folder)

    with open(f"../{sub_folder}/output.txt", "a") as file:
        file.write(f"\nAll Test results - iccma and generated\n")
    
    (f1_all, accuracy_all, precision_all, recall_all) = evaluate_model(test_all_loader, model, 0, sub_folder, test_set=True)
    bootstrap_all = calculate_bootstrap_confidence_intervals(test_set_all, model, sub_folder)

    # Store results in a dictionary
    results = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'bootstrap_25_100': bootstrap_25_100,
        'bootstrap_iccma': bootstrap_iccma,
        'bootstrap_all': bootstrap_all
    }
    
    # Plot the single training run metrics and evaluation statistics
    plot_metrics(results, sub_folder)
    return results


def run_training(model_type):
    """Modified run_training function to call the multi-run version"""
    # Set a master seed for overall reproducibility
    set_seeds(42)
    if model_type == 'gcn':
        sub_folder = "results_final_gcn"
    elif model_type == 'gat':
        sub_folder = "results_final_gat"
    else:
        raise ValueError("model_type must be either 'gcn' or 'gat'")
    
    # Run multiple training sessions and average results
    results = train_and_evaluate(model_type=model_type, epochs=250, sub_folder=sub_folder)
    
    # You could also save the average results to a file
    np.save(f'../{sub_folder}/training_results.npy', results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GCN or GAT model on ABA datasets.")
    parser.add_argument('--model_type', type=str, required=False, default='gcn', choices=['gcn', 'gat'], help="Model type: 'gcn' or 'gat' (default: gcn)")
    parser.add_argument('--epochs', type=int, required=False, default=250, help="Number of training epochs (default: 250)")
    parser.add_argument('--output_folder', type=str, required=False, default=None, help="Output folder for results (default: results_final_gcn or results_final_gat)")
    args = parser.parse_args()

    model_type = args.model_type
    epochs = args.epochs
    # Set output folder based on model type if not provided
    if args.output_folder is not None:
        sub_folder = args.output_folder
    else:
        sub_folder = "results_final_gcn" if model_type == 'gcn' else "results_final_gat"

    if model_type == 'gcn':
        ### Optimized hyperparameters for GCN
        PATIENCE = 10
        BATCH_SIZE = 128
        POS_WEIGHT = 2.25
        NUM_WORKERS = 0
        THRESHOLD = 0.4998769870504005
        EMBEDDING_DIM = 32
        HIDDEN_DIM = 32
        NUM_LAYERS = 10
        DROPOUT = 0.02943105695360959
        LEARNING_RATE = 0.008622415975216019
    elif model_type == 'gat':
        ### Optimized hyperparameters for GAT
        PATIENCE = 30
        BATCH_SIZE = 64
        POS_WEIGHT = 1.75
        NUM_WORKERS = 0
        THRESHOLD = 0.4578899746620344
        EMBEDDING_DIM = 64
        HIDDEN_DIM = 64
        NUM_LAYERS = 10
        DROPOUT = 0.2198191427741004
        LEARNING_RATE = 0.006584746811018268
    else:
        raise ValueError("model_type must be either 'gcn' or 'gat'")

    run_training(model_type=model_type)