import os
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from GCN_learnable import GCNLearnableModel
from GAT_learnable import GATLearnableModel


# Trainer class to handle model training and evaluation
class HyperParamTrainer:
    def __init__(self, run, config=None):
        """
        Args:
            config: Configuration dictionary for the run
            project_name: Name of the W&B project
        """
        # Initialize W&B
        self.run = run
        self.config = config
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        print(f"Using device: {self.device}")
        
        # Set up the model
        if config.model == 'gcn':
            self.model = GCNLearnableModel(
                in_features=config.in_features,
                hidden_features=config.hidden_dim,
                out_features=config.out_features,
                embedding_dim=config.embedding_dim,
                num_layers=config.num_layers,
                dropout=config.dropout
            ).to(self.device)
        elif config.model == 'gat':
            self.model = GATLearnableModel(
                in_features=config.in_features,
                hidden_features=config.hidden_dim,
                out_features=config.out_features,
                embedding_dim=config.embedding_dim,
                num_layers=config.num_layers,
                dropout=config.dropout
            ).to(self.device)
        else:
            raise ValueError("Model must be either 'gcn' or 'gat'")
        
        # Set up optimizer
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Set up learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=config.lr_patience,
            verbose=True
        )
        
        # Loss function
        pos_weight = torch.tensor([config.pos_weight]).to(device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # For tracking best model
        self.best_model_path = f"best_model_run_{self.run.id}.pt"
        
        # Initialize counters for early stopping
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_val_precision = 0
        
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        total_nodes = 0
        
        for batched_graph in train_loader:
            batched_graph = batched_graph.to(self.device)

            inputs = {
                'assmpt': batched_graph.nodes['assmpt'].data['features'],
                'rule': batched_graph.nodes['rule'].data['features'],
                'non_assmpt': batched_graph.nodes['non_assmpt'].data['features']
            }

            labels = batched_graph.nodes['assmpt'].data['label']

            outputs = self.model(batched_graph, inputs)
            logits = outputs['assmpt'].squeeze(1)
            
            loss = self.criterion(logits, labels.float())

            # Zero out all of the gradients for the variables which the optimizer will update.
            self.optimizer.zero_grad()
            loss.backward()
            # Update the parameters of the model using the gradients
            self.optimizer.step()

            # Weight the loss by number of nodes in this graph
            num_nodes = labels.size(0)
            epoch_loss += (loss * num_nodes).item()
            total_nodes += num_nodes
        
        avg_epoch_loss = epoch_loss / total_nodes

        return {"loss": avg_epoch_loss}
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_examples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batched_graph in dataloader:
                batched_graph = batched_graph.to(self.device)

                inputs = {
                    'assmpt': batched_graph.nodes['assmpt'].data['features'],
                    'rule': batched_graph.nodes['rule'].data['features'],
                    'non_assmpt': batched_graph.nodes['non_assmpt'].data['features']
                }

                batch_labels = batched_graph.nodes['assmpt'].data['label']

                outputs = self.model(batched_graph, inputs)
                logits = outputs['assmpt'].squeeze(1)

                loss = self.criterion(logits, batch_labels.float())
                total_loss += (loss * batch_labels.size(0)).item()

                predictions = (torch.sigmoid(logits) > self.config.threshold).long().detach().cpu().numpy()
                labels = batch_labels.detach().cpu().numpy()

                all_preds.extend(predictions)
                all_labels.extend(labels)

                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += len(labels)
            
        accuracy = total_correct / total_examples
        avg_loss = total_loss / total_examples

        # Compute precision, recall, and F1
        # For binary classification, these default to 'binary' if your labels are in {0, 1}.
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def train(self, train_loader, val_loader, fold):
        """Full training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)

            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, ")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Early stopping based on validation loss
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_f1 = val_metrics["f1"]
                self.best_val_accuracy = val_metrics["accuracy"]
                self.best_val_recall = val_metrics["recall"]
                self.best_val_precision = val_metrics["precision"]

                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                wandb.save(self.best_model_path)
                print(f"Saved new best model with validation loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"Increasing patience")
            
            # Log metrics
            self.run.log({
                f"fold-{fold}_train_loss": train_metrics["loss"],
                f"fold-{fold}_val_loss": val_metrics["loss"],
                f"fold-{fold}_val_accuracy": val_metrics["accuracy"],
                f"fold-{fold}_val_f1": val_metrics["f1"],
                f"fold-{fold}_val_precision": val_metrics["precision"],
                f"fold-{fold}_val_recall": val_metrics["recall"],
                f"fold-{fold}_learning_rate": current_lr
            })
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}, "
                  f"LR: {current_lr:.6f}")
                
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Create model artifact
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            description=f"Model with val_f1={self.best_val_loss:.4f}"
        )
        artifact.add_file(self.best_model_path)
        self.run.log_artifact(artifact)

        return (self.best_val_f1, self.best_val_accuracy, self.best_val_recall, self.best_val_precision)
        


