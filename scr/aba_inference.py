from GCN_learnable import GCNLearnableModel
from GAT_learnable import GATLearnableModel
import torch 

DEFAULTS = {
    'gcn': {
        'threshold': 0.5,
        'hidden_features': 32,
        'embedding_dim': 32,
        'num_layers': 10,
        'dropout': 0.02943105695360959,
    },
    'gat': {
        'threshold': 0.45,
        'hidden_features': 64,
        'embedding_dim': 64,
        'num_layers': 10,
        'dropout': 0.2198191427741004,
        'num_heads': 4,  # if needed
    }
}

USE_GPU = True
dtype = torch.float32 
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)

class ABAInferenceEngine:
    """
    Standalone inference engine for ABA (Assumption-Based Argumentation) frameworks.
    Supports both GCN and GAT models.
    """
    
    def __init__(self, model_type, model_path, inclusion_threshold=None):
        """
        Initialize the inference engine.
        Args:
            model_type (str): 'gcn' or 'gat'
            model_path (str): Path to the trained model file
            inclusion_threshold (float): Optional threshold for acceptance (overrides model default)
        """
        assert model_type in ['gcn', 'gat'], "model_type must be 'gcn' or 'gat'"
        self.model_type = model_type
        params = DEFAULTS[model_type]
        if inclusion_threshold is not None:
            self.threshold = inclusion_threshold
        else:
            self.threshold = params['threshold']
        self.enumeration_threshold = self.threshold
        
        # Model hyperparameters (should match training configuration)
        if model_type == 'gcn':
            self.model = GCNLearnableModel(
                in_features=2, 
                hidden_features=params['hidden_features'], 
                out_features=1, 
                embedding_dim=params['embedding_dim'], 
                num_layers=params['num_layers'], 
                dropout=params['dropout']
            )
        elif model_type == 'gat':
            self.model = GATLearnableModel(
                in_features=2, 
                hidden_features=params['hidden_features'], 
                out_features=1, 
                embedding_dim=params['embedding_dim'], 
                num_layers=params['num_layers'], 
                dropout=params['dropout']
            )
        
        # Load trained model
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        print(f"Model ({model_type}) loaded successfully on {device}")
    
    def predict(self, hetero_graph):
        """
        Make predictions on a heterogeneous graph.
        Args:
            hetero_graph: DGL heterogeneous graph
        Returns:
            torch.Tensor: Probabilities for each assumption being accepted
        """
        hetero_graph = hetero_graph.to(device)
        with torch.no_grad():
            inputs = {
                'assmpt': hetero_graph.nodes['assmpt'].data['features'],
                'rule': hetero_graph.nodes['rule'].data['features'],
                'non_assmpt': hetero_graph.nodes['non_assmpt'].data['features']
            }
            outputs = self.model(hetero_graph, inputs)
            logits = outputs['assmpt'].squeeze(1)
            probabilities = torch.sigmoid(logits)
        return probabilities.cpu()
    
    def inference(self, hetero_graph, assmpt_mapping):
        """
        Perform complete inference on an ABA file.
        Args:
            hetero_graph: DGL heterogeneous graph
            assmpt_mapping: Mapping of assumption names to indices
        Returns:
            list: List of tuples (assumption_name, probability, accepted) 
                  sorted by probability (highest first)
        """
        try:
            # Make predictions
            probabilities = self.predict(hetero_graph)
            # Create reverse mapping (index -> assumption name)
            reverse_mapping = {v: k for k, v in assmpt_mapping.items()}
            # Combine results
            results = []
            for idx, prob in enumerate(probabilities):
                assumption_name = reverse_mapping[idx]
                accepted = prob.item() > self.threshold
                results.append((assumption_name, prob.item(), accepted))
            # Sort by probability (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"\nInference Results (threshold: {self.threshold}):")
            print("-" * 60)
            for assumption, prob, accepted in results:
                status = "ACCEPTED" if accepted else "REJECTED"
                print(f"{assumption:<20} | {prob:.6f} | {status}")
            accepted_count = sum(1 for _, _, accepted in results if accepted)
            print(f"\nSummary: {accepted_count}/{len(results)} assumptions accepted")
            # Return all results for further filtering
            return results
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise