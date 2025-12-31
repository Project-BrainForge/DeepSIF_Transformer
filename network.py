import torch
from torch import nn
import torch.nn.functional as F
import math


class ImprovedTransformerLayer(nn.Module):
    """Improved transformer layer with pre-norm and better regularization"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Pre-norm architecture (more stable)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Improved feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # Better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout * 0.5)  # Lower dropout on output
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feedforward
        norm_x = self.norm2(x)
        ff_out = self.feedforward(norm_x)
        x = x + ff_out
        
        return x


class MLPSpatialFilter(nn.Module):

    def __init__(self, num_sensor, num_hidden, activation):
        super(MLPSpatialFilter, self).__init__()
        self.fc11 = nn.Linear(num_sensor, num_sensor)
        self.fc12 = nn.Linear(num_sensor, num_sensor)
        self.fc21 = nn.Linear(num_sensor, num_hidden)
        self.fc22 = nn.Linear(num_hidden, num_hidden)
        self.fc23 = nn.Linear(num_sensor, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        x = self.activation(self.fc12(self.activation(self.fc11(x))) + x)
        x = self.activation(self.fc22(self.activation(self.fc21(x))) + self.fc23(x))
        out['value'] = self.value(x)
        out['value_activation'] = self.activation(out['value'])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTemporalFilter(nn.Module):
    
    def __init__(self, input_size, num_source, num_layer, activation, 
                 d_model=256, nhead=8, dropout=0.15):
        super(TransformerTemporalFilter, self).__init__()
        
        self.input_size = input_size
        self.num_source = num_source
        self.num_layer = num_layer
        self.d_model = d_model
        
        # Input projection with layer norm and residual connection
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Learnable positional encoding (better than sinusoidal for our case)
        self.pos_embedding = nn.Parameter(torch.randn(1000, d_model) * 0.1)
        
        # Pre-LayerNorm Transformer encoder (more stable training)
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerLayer(d_model, nhead, d_model * 2, dropout)
            for _ in range(num_layer)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection with residual connection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_source)
        )
        
        self.activation = nn.__dict__[activation]() if activation in nn.__dict__ else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization for better gradient flow"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        out = dict()
        
        # Input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions with residual-like connection
        projected = self.input_projection(x)  # (batch_size, seq_len, d_model)
        projected = self.input_norm(projected)
        projected = self.activation(projected)
        
        # Add learnable positional encoding
        if seq_len <= self.pos_embedding.size(0):
            pos_enc = self.pos_embedding[:seq_len].unsqueeze(0)
            x = projected + pos_enc
        else:
            # Handle sequences longer than max position
            pos_enc = self.pos_embedding.unsqueeze(0).repeat(1, (seq_len // 1000) + 1, 1)[:, :seq_len]
            x = projected + pos_enc
        
        # Apply input dropout
        x = self.dropout(x)
        
        # Apply improved transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to output dimension with skip connection to projected input
        output = self.output_projection(x)  # (batch_size, seq_len, num_source)
        
        out['transformer'] = output
        return out


# Keep the original LSTM-based temporal filter for compatibility
class TemporalFilter(nn.Module):

    def __init__(self, input_size, num_source, num_layer, activation):
        super(TemporalFilter, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size, num_source, batch_first=True, num_layers=num_layer))
        self.num_layer = num_layer
        self.input_size = input_size
        self.activation = nn.__dict__[activation]()

    def forward(self, x):
        out = dict()
        # c0/h0 : num_layer, T, num_out
        for l in self.rnns:
            l.flatten_parameters()
            x, _ = l(x)

        out['rnn'] = x  # seq_len, batch, num_directions * hidden_size
        return out


class TemporalInverseNet(nn.Module):

    def __init__(self, num_sensor=75, num_source=994, rnn_layer=3,
                 spatial_model=MLPSpatialFilter, temporal_model=TemporalFilter,
                 spatial_output='value_activation', temporal_output='rnn',
                 spatial_activation='ELU', temporal_activation='ELU', temporal_input_size=500):
        super(TemporalInverseNet, self).__init__()
        self.attribute_list = [num_sensor, num_source, rnn_layer,
                               spatial_model, temporal_model, spatial_output, temporal_output,
                               spatial_activation, temporal_activation, temporal_input_size]
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output
        # Spatial filtering
        self.spatial = spatial_model(num_sensor, temporal_input_size, spatial_activation)
        # Temporal filtering
        self.temporal = temporal_model(temporal_input_size, num_source, rnn_layer, temporal_activation)

    def forward(self, x):
        out = dict()
        out['fc2'] = self.spatial(x)[self.spatial_output]
        x = out['fc2']
        out['last'] = self.temporal(x)[self.temporal_output]
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerTemporalInverseNet(nn.Module):
    """
    Optimized version of TemporalInverseNet using improved Transformer
    """
    
    def __init__(self, num_sensor=75, num_source=994, transformer_layers=4,
                 spatial_model=MLPSpatialFilter, temporal_model=TransformerTemporalFilter,
                 spatial_output='value_activation', temporal_output='transformer',
                 spatial_activation='GELU', temporal_activation='GELU', temporal_input_size=500,
                 d_model=256, nhead=8, dropout=0.15):
        super(TransformerTemporalInverseNet, self).__init__()
        
        self.attribute_list = [num_sensor, num_source, transformer_layers,
                               spatial_model, temporal_model, spatial_output, temporal_output,
                               spatial_activation, temporal_activation, temporal_input_size,
                               d_model, nhead, dropout]
        
        self.spatial_output = spatial_output
        self.temporal_output = temporal_output
        
        # Enhanced spatial filtering with batch norm
        self.spatial = spatial_model(num_sensor, temporal_input_size, spatial_activation)
        self.spatial_dropout = nn.Dropout(dropout * 0.5)
        self.spatial_norm = nn.LayerNorm(temporal_input_size)
        
        # Temporal filtering with improved Transformer
        self.temporal = temporal_model(
            temporal_input_size, num_source, transformer_layers, temporal_activation,
            d_model, nhead, dropout
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        out = dict()
        
        # Spatial filtering with regularization
        spatial_out = self.spatial(x)[self.spatial_output]
        spatial_out = self.spatial_norm(spatial_out)
        spatial_out = self.spatial_dropout(spatial_out)
        out['fc2'] = spatial_out
        
        # Temporal filtering with improved Transformer
        temporal_out = self.temporal(spatial_out)[self.temporal_output]
        out['last'] = temporal_out
        
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
