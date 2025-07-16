import dgl
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

# class LearnableGraphEmbedder(nn.Module):
#     def __init__(self, in_features, embedding_dim=64):
#         super().__init__()
#         # Learnable transformation layer
#         self.embedding_layer = nn.Linear(in_features, embedding_dim)
    
#     def forward(self, features):
#         # Transform input features to learnable embeddings
#         return self.embedding_layer(features)

class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, embedding_dim=64):
        super().__init__()

        # self.atom_embedder = LearnableGraphEmbedder(in_features, embedding_dim)
        # self.rule_embedder = LearnableGraphEmbedder(in_features, embedding_dim)

        self.conv1 = dglnn.HeteroGraphConv({
            ('assmpt', 'supports', 'rule'): dglnn.GraphConv(in_features, hidden_features),
            ('non_assmpt', 'supports', 'rule'): dglnn.GraphConv(in_features, hidden_features),
            ('non_assmpt', 'attacks', 'assmpt'): dglnn.GraphConv(in_features, hidden_features),
            ('assmpt', 'attacks', 'assmpt'): dglnn.GraphConv(in_features, hidden_features),
            ('rule', 'derives', 'non_assmpt'): dglnn.GraphConv(in_features, hidden_features),
            ('rule', 'derives', 'assmpt'): dglnn.GraphConv(in_features, hidden_features),
            ('assmpt', 'supports', 'assmpt'): dglnn.GraphConv(in_features, hidden_features),
            ('rule', 'supports', 'rule'): dglnn.GraphConv(in_features, hidden_features),
            ('non_assmpt', 'supports', 'non_assmpt'): dglnn.GraphConv(in_features, hidden_features)
        }, aggregate='sum')

        self.conv2 = dglnn.HeteroGraphConv({
            ('assmpt', 'supports', 'rule'): dglnn.GraphConv(hidden_features, hidden_features),
            ('non_assmpt', 'supports', 'rule'): dglnn.GraphConv(hidden_features, hidden_features),
            ('non_assmpt', 'attacks', 'assmpt'): dglnn.GraphConv(hidden_features, hidden_features),
            ('assmpt', 'attacks', 'assmpt'): dglnn.GraphConv(hidden_features, hidden_features),
            ('rule', 'derives', 'non_assmpt'): dglnn.GraphConv(hidden_features, hidden_features),
            ('rule', 'derives', 'assmpt'): dglnn.GraphConv(hidden_features, hidden_features),
            ('assmpt', 'supports', 'assmpt'): dglnn.GraphConv(hidden_features, hidden_features),
            ('rule', 'supports', 'rule'): dglnn.GraphConv(hidden_features, hidden_features),
            ('non_assmpt', 'supports', 'non_assmpt'): dglnn.GraphConv(hidden_features, hidden_features)
        }, aggregate='sum')

        # Linear layer for classification
        self.classifier = nn.Linear(hidden_features, out_features)

        self._init_weights()
    
    def _init_weights(self):
        # Initialize GraphConv layers in conv1
        for rel, conv in self.conv1.mods.items():
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        # Initialize GraphConv layers in conv2
        for rel, conv in self.conv2.mods.items():
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


    def forward(self, graph, inputs):
        # Learn embeddings for atoms and rules
        # atom_embeds = self.atom_embedder(inputs['a'])
        # rule_embeds = self.rule_embedder(inputs['r'])
        
        # # Prepare inputs with learned embeddings
        # learned_inputs = {
        #     'a': atom_embeds,
        #     'r': rule_embeds
        # }

        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k,v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k,v in h.items()}
        h = {k: self.classifier(v) for k,v in h.items()}
        return h

if __name__ == "__main__":
    graphs, _ = dgl.load_graphs('../non_flat_hetero_graph.bin')
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

    model = GCNModel(2, 10, 1)

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
