import torch
from torch_geometric.nn import GraphConv, GATv2Conv

class GCN_inner(torch.nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, heads, concat, dropout):
        super(GCN_inner, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_layers, aggr = 'mean')
        self.bn1 = torch.nn.BatchNorm1d(hidden_layers)
        self.conv2 = GATv2Conv(hidden_layers, out_channels, heads = heads, 
                               concat = concat, dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
class GCN_inner_image(torch.nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, heads, concat, dropout):
        super(GCN_inner_image, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_layers, aggr = 'mean')
        self.bn1 = torch.nn.BatchNorm1d(hidden_layers)
        self.conv2 = GATv2Conv(hidden_layers, out_channels, heads = heads, concat=concat, 
                               dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x