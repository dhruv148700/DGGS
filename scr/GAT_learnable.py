import dgl
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class LearnableGraphEmbedder(nn.Module):
    def __init__(self, in_features, embedding_dim=64):
        super().__init__()
        # Learnable transformation layer
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
    
    def forward(self, features):
        # Transform input features to learnable embeddings
        return self.embedding_layer(features)

class GATLearnableModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads=4, embedding_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.assmpt_embedder = LearnableGraphEmbedder(in_features, embedding_dim)
        self.rule_embedder = LearnableGraphEmbedder(in_features, embedding_dim)
        self.non_assmpt_embedder = LearnableGraphEmbedder(in_features, embedding_dim)


        self.convs = nn.ModuleList()
        self.edge_types = [
            ('assmpt', 'supports', 'rule'),
            ('non_assmpt', 'supports', 'rule'),
            ('non_assmpt', 'attacks', 'assmpt'),
            ('assmpt', 'attacks', 'assmpt'),
            ('rule', 'derives', 'non_assmpt'),
            ('rule', 'derives', 'assmpt'),
            ('assmpt', 'supports', 'assmpt'),
            ('rule', 'supports', 'rule'),
            ('non_assmpt', 'supports', 'non_assmpt')
        ]

        conv_dict = {edge_type: dglnn.GATConv(embedding_dim, hidden_features // num_heads, num_heads, allow_zero_in_degree=True) 
                     for edge_type in self.edge_types}
        self.convs.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

        for _ in range(num_layers - 2):
            conv_dict = {edge_type: dglnn.GATConv(hidden_features, hidden_features // num_heads, num_heads, allow_zero_in_degree=True) 
                         for edge_type in self.edge_types}
            self.convs.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))
        
        # Last layer (if more than one layer): hidden_features -> hidden_features
        if num_layers > 1:
            conv_dict = {edge_type: dglnn.GATConv(hidden_features, hidden_features // num_heads, num_heads, allow_zero_in_degree=True)
                         for edge_type in self.edge_types}
            self.convs.append(dglnn.HeteroGraphConv(conv_dict, aggregate='sum'))

        # Classifier for each node type
        self.classifiers = nn.ModuleDict({
            'assmpt': nn.Linear(hidden_features, out_features),
            'rule': nn.Linear(hidden_features, out_features),
            'non_assmpt': nn.Linear(hidden_features, out_features)
        })
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleDict({
            'assmpt': nn.LayerNorm(hidden_features),
            'rule': nn.LayerNorm(hidden_features),
            'non_assmpt': nn.LayerNorm(hidden_features)
        })

        self._init_weights()
    
    def _init_weights(self):
         # Initialize embedders
        for layer in [self.assmpt_embedder, self.rule_embedder, self.non_assmpt_embedder]:
            nn.init.xavier_uniform_(layer.embedding_layer.weight)
            nn.init.zeros_(layer.embedding_layer.bias)
        
        # Initialize GAT convolutional layers
        for conv in self.convs:
            for edge_module in conv.mods.values():
                # For each GATConv module
                if hasattr(edge_module, 'fc'):
                    nn.init.xavier_uniform_(edge_module.fc.weight)
                    if edge_module.fc.bias is not None:
                        nn.init.zeros_(edge_module.fc.bias)
                
                # Initialize attention weights
                if hasattr(edge_module, 'attn_l'):
                    nn.init.xavier_uniform_(edge_module.attn_l)
                if hasattr(edge_module, 'attn_r'):
                    nn.init.xavier_uniform_(edge_module.attn_r)
        
        # Initialize classifier
        for _, layer in self.classifiers.items():
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


    def forward(self, graph, inputs):
        # Learn embeddings for atoms and rules
        assmpt_embeds = self.assmpt_embedder(inputs['assmpt'])
        rule_embeds = self.rule_embedder(inputs['rule'])
        non_assmpt_embeds = self.non_assmpt_embedder(inputs['non_assmpt'])
        
        # Prepare inputs with learned embeddings
        h = {
            'assmpt': assmpt_embeds,
            'non_assmpt': non_assmpt_embeds,
            'rule': rule_embeds
        }

        # Apply convolutional layers with ReLU and dropout in between
        for i, conv in enumerate(self.convs):
            h_new = conv(graph, h)
            # Apply layer normalization, ReLU, and dropout between layers
            h_new = {k: self.layer_norms[k](v.flatten(1)) for k, v in h_new.items()}
            h_new = {k: F.relu(v) for k, v in h_new.items()}
            h_new = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_new.items()}
            
            # Residual connection if dimensions match
            if i > 0 and self.num_layers > 3:  # Only for layers after the first, as dimensions need to match
                h = {k: h_new[k] + h[k] for k in h}
            else:
                h = h_new

        # Apply classifiers
        output = {k: self.classifiers[k](v) for k, v in h.items()}
        return output

if __name__ == "__main__":
    graphs, _ = dgl.load_graphs('../hetero_graph_generated2.bin')
    # Retrieve the first graph (since we saved one graph)
    hetero_graph = graphs[0]

    print(hetero_graph)
    print(hetero_graph.ntypes)
    print(hetero_graph.canonical_etypes)
    # print(hetero_graph.edges(etype='derives'))
    # print(hetero_graph.edges(etype='attacks'))
    print(hetero_graph.nodes['assmpt'].data['features'])
    print(hetero_graph.nodes['assmpt'].data['label'])
    label_length = len(hetero_graph.nodes['assmpt'].data['label'])

    model = GATLearnableModel(2, 128, 1, num_layers=4)

    inputs = {
        'assmpt': hetero_graph.nodes['assmpt'].data['features'].float(),
        'non_assmpt': hetero_graph.nodes['non_assmpt'].data['features'].float(),
        'rule': hetero_graph.nodes['rule'].data['features'].float()
    }
    print(inputs)

    with torch.no_grad():
        output = model(hetero_graph, inputs)
    
    print("Outputs:")
    print(output)
    print("correct num of labels:", label_length == len(output['assmpt']))
