import numpy as np
import matplotlib.pyplot as plt 

def plot_metrics(results, sub_folder):
    """
    Plot average metrics across multiple runs with standard deviation bands.
    
    Args:
        results: Dictionary containing metrics
        epochs: Number of training epochs
        num_runs: Number of training runs performed
    """
    train_loss = results['train_loss']
    val_loss = results['val_loss']
    mean_test_f1_1 = results['bootstrap_25_100']['f1'][0]
    std_test_f1_1 = results['bootstrap_25_100']['f1'][1]
    mean_test_f1_2 = results['bootstrap_iccma']['f1'][0]
    std_test_f1_2 = results['bootstrap_iccma']['f1'][1]

    epoch_range = range(1, len(train_loss) + 1)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot average training loss with std deviation band
    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color1)

    # Mean line
    ax1.plot(epoch_range, train_loss, '-', color=color1, 
             linewidth=2, label=f'Training Loss')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create a second y-axis for validation accuracy
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Validation Loss', color=color2)
    
    # Mean line
    ax2.plot(epoch_range, val_loss, '-', color=color2, 
             linewidth=2, label=f'Validation Loss')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add grid and title
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Training Metrics For Run')
    
    # Create custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add test f1 annotation
    test_f1_text = f"Test F1 ICCMA (25/100 element samples): {mean_test_f1_1:.4f} ± {std_test_f1_1:.4f}\n Test F1 ICCMA all samples: {mean_test_f1_2:.4f} ± {std_test_f1_2:.4f}"
    if "bootstrap_all" in results:
        mean_test_f1_3 = results['bootstrap_all']['f1'][0]
        std_test_f1_3 = results['bootstrap_all']['f1'][1]
        test_f1_text += f"\n Test F1 all samples: {mean_test_f1_3:.4f} ± {std_test_f1_3:.4f}"

    plt.annotate(test_f1_text, xy=(0.5, 0.02), xycoords='figure fraction', 
                 ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", 
                                                    facecolor='white', alpha=0.8))
    
    # Save and show
    fig.tight_layout()
    plt.savefig(f'../{sub_folder}/average_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cross_validation_metrics(results, k, sub_folder):
    """
    Plot training and validation metrics for cross-validation.
    
    Args:
        results: Dictionary containing results from cross-validation
        k: Number of folds
        sub_folder: Output directory for saving plots
    """
    
    # Plot individual learning curves for each fold
    for fold in range(k):
        plt.figure(figsize=(10, 6))
        
        train_losses = results['fold_train_losses'][fold]
        val_losses = results['fold_val_losses'][fold]
        epochs = np.arange(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.title(f'Fold {fold+1} Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'../{sub_folder}/fold_{fold+1}_learning_curves.png')
        plt.close()
    
    # Create a figure for the aggregated learning curves (as before)
    plt.figure(figsize=(12, 6))
    
    # Calculate max length of training epochs (might differ across folds due to early stopping)
    max_epochs = max([len(losses) for losses in results['fold_train_losses']])
    
    # Prepare arrays for averaging
    all_train_losses = np.zeros((k, max_epochs))
    all_val_losses = np.zeros((k, max_epochs))
    
    # Fill arrays with data, padding with NaN for missing epochs
    for i in range(k):
        train_len = len(results['fold_train_losses'][i])
        val_len = len(results['fold_val_losses'][i])
        
        all_train_losses[i, :train_len] = results['fold_train_losses'][i]
        all_train_losses[i, train_len:] = np.nan
        
        all_val_losses[i, :val_len] = results['fold_val_losses'][i]
        all_val_losses[i, val_len:] = np.nan
    
    # Calculate mean and std ignoring NaN values
    train_mean = np.nanmean(all_train_losses, axis=0)
    train_std = np.nanstd(all_train_losses, axis=0)
    val_mean = np.nanmean(all_val_losses, axis=0)
    val_std = np.nanstd(all_val_losses, axis=0)
    
    # Plot epochs
    epochs = np.arange(1, max_epochs + 1)
    
    # Plot training loss
    plt.plot(epochs, train_mean, 'b-', label='Training Loss')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='b')
    
    # Plot validation loss
    plt.plot(epochs, val_mean, 'r-', label='Validation Loss')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='r')
    
    plt.title(f'{k}-Fold Cross-Validation Learning Curves (Aggregated)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'../{sub_folder}/cross_validation_learning_curves_aggregated.png')
    
    # Create a figure for comparing all fold learning curves
    plt.figure(figsize=(14, 10))
    
    # Plot training loss for each fold
    plt.subplot(2, 1, 1)
    for fold in range(k):
        train_losses = results['fold_train_losses'][fold]
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f'Fold {fold+1}')
    
    plt.title('Training Loss Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation loss for each fold
    plt.subplot(2, 1, 2)
    for fold in range(k):
        val_losses = results['fold_val_losses'][fold]
        epochs = np.arange(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, label=f'Fold {fold+1}')
    
    plt.title('Validation Loss Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../{sub_folder}/cross_validation_all_folds_comparison.png')
    
    # Create a figure for the F1 scores (as before)
    plt.figure(figsize=(10, 6))
    
    # Plot individual fold F1 scores
    plt.bar(range(1, k + 1), results['fold_test_f1s'], color='skyblue')
    plt.axhline(y=results['mean_test_f1'], color='r', linestyle='-', label=f'Mean F1: {results["mean_test_f1"]:.4f}')
    
    # Add error bands for ±1 std
    plt.axhline(y=results['mean_test_f1'] + results['std_test_f1'], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_test_f1'] - results['std_test_f1'], color='r', linestyle='--', alpha=0.5)
    plt.fill_between([0.5, k + 0.5], 
                     [results['mean_test_f1'] - results['std_test_f1']] * 2, 
                     [results['mean_test_f1'] + results['std_test_f1']] * 2, 
                     color='r', alpha=0.1)
    
    plt.title(f'{k}-Fold Cross-Validation F1 Scores')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, k + 1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f'../{sub_folder}/cross_validation_f1_scores.png')
    plt.close('all')
