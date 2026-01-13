import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.io import loadmat


# ============================================================================
# ELECTRODE TOPOLOGY & 2D GRID UTILITIES (for CNN Spatial Filter)
# ============================================================================

def load_electrode_montage(electrode_file='anatomy/electrode_75.mat'):
    """
    Load electrode positions from EEGLAB format mat file using MNE.
    
    Args:
        electrode_file: Path to electrode_75.mat file
        
    Returns:
        montage: MNE DigMontage object with 75 electrode positions
        
    Raises:
        ImportError: If MNE is not installed
        FileNotFoundError: If electrode file not found
    """
    try:
        import mne
        from mne.channels import make_dig_montage
    except ImportError:
        raise ImportError("MNE is required for electrode topology. Install with: pip install mne")
    
    # Load EEGLAB electrode format
    mat_data = loadmat(electrode_file)
    eloc75 = mat_data['eloc75']  # EEGLAB format electrode locations, shape (1, 75)
    
    # Handle structured array (EEGLAB format is structured)
    if eloc75.dtype.names:
        # Structured array with shape (1, 75)
        # Extract X, Y, Z coordinates - each field contains 75 nested arrays
        coords = np.zeros((75, 3), dtype=np.float32)
        
        # Extract coordinates from fields X, Y, Z
        for i in range(75):
            # Access nested arrays: eloc75['field'][0, i] gives the array for electrode i
            x_val = eloc75['X'][0, i]
            y_val = eloc75['Y'][0, i]
            z_val = eloc75['Z'][0, i]
            
            # Extract scalar from nested array structure
            coords[i, 0] = float(x_val.flat[0]) if x_val.size > 0 else 0.0
            coords[i, 1] = float(y_val.flat[0]) if y_val.size > 0 else 0.0
            coords[i, 2] = float(z_val.flat[0]) if z_val.size > 0 else 0.0
    else:
        # Regular array - assume shape (n_elec, 3) or (3, n_elec)
        if eloc75.shape[0] == 3:
            coords = eloc75.T.astype(np.float32)
        else:
            coords = eloc75[:, :3].astype(np.float32)
    
    # Detect if coordinates are spherical angles (degrees) or Cartesian
    coord_max = np.max(np.abs(coords))
    if coord_max > 10:  # Likely spherical angles in degrees
        # Convert spherical (theta, phi, radius) to Cartesian
        # Using scipy's spherical to Cartesian conversion
        from scipy.spatial.transform import Rotation
        theta = np.radians(coords[:, 0])  # Azimuth in radians
        phi = np.radians(coords[:, 1])    # Elevation in radians
        
        # Spherical to Cartesian: x = r*sin(phi)*cos(theta), y = r*sin(phi)*sin(theta), z = r*cos(phi)
        r = np.ones(len(coords))  # Normalize to unit sphere
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        coords_3d = np.column_stack([x, y, z])
    else:
        # Already in Cartesian coordinates
        coords_3d = coords
    
    # Create channel names
    ch_names = [f'E{i+1}' for i in range(75)]
    
    # Create MNE montage
    montage = make_dig_montage(
        ch_pos=dict(zip(ch_names, coords_3d)),
        coord_frame='head'
    )
    
    return montage


def project_electrodes_to_2d(montage, grid_size=64):
    """
    Project 3D electrode positions to 2D stereographic projection for CNN grid.
    
    Args:
        montage: MNE DigMontage object
        grid_size: Size of output 2D grid (default 64x64)
        
    Returns:
        grid_pos: (75, 2) array of electrode positions in grid coordinates
    """
    positions = montage.get_positions()
    ch_pos = positions['ch_pos']
    coords_3d = np.array([ch_pos[name] for name in montage.ch_names])
    
    # Normalize to unit sphere
    r = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    coords_norm = coords_3d / r
    
    # Stereographic projection: (x,y,z) -> (x/(1-z), y/(1-z))
    z = coords_norm[:, 2]
    factor = 1.0 / (1.0 - z + 1e-6)  # Avoid division by zero
    x_2d = coords_norm[:, 0] * factor
    y_2d = coords_norm[:, 1] * factor
    
    # Normalize to grid range [0, grid_size-1]
    pos_2d = np.column_stack([x_2d, y_2d])
    pos_2d_min = pos_2d.min(axis=0)
    pos_2d_max = pos_2d.max(axis=0)
    pos_2d_range = pos_2d_max - pos_2d_min
    
    pos_2d = (pos_2d - pos_2d_min) / (pos_2d_range + 1e-6) * (grid_size - 1)
    
    return pos_2d


def create_electrode_grid_mapping(electrode_file='anatomy/electrode_75.mat', grid_size=64):
    """
    Create electrode-to-grid mapping using MNE electrode topology.
    
    Args:
        electrode_file: Path to electrode_75.mat
        grid_size: Size of output 2D grid
        
    Returns:
        grid_pos: (75, 2) electrode positions in grid
        electrode_indices: (grid_size, grid_size) array with electrode indices (-1 for empty)
    """
    montage = load_electrode_montage(electrode_file)
    grid_pos = project_electrodes_to_2d(montage, grid_size=grid_size)
    
    # Create grid with electrode indices
    electrode_indices = np.full((grid_size, grid_size), -1, dtype=np.int16)
    for idx, (x, y) in enumerate(grid_pos):
        x_int = int(np.clip(np.round(x), 0, grid_size-1))
        y_int = int(np.clip(np.round(y), 0, grid_size-1))
        electrode_indices[y_int, x_int] = idx
    
    return grid_pos, electrode_indices


def eeg_to_2d_grid(eeg_data, grid_pos, grid_size=64, interpolate=False):
    """
    Convert EEG data from (time_steps, 75) to (time_steps, grid_size, grid_size).
    
    Args:
        eeg_data: (time_steps, 75) EEG array
        grid_pos: (75, 2) electrode positions in grid from project_electrodes_to_2d()
        grid_size: Size of output 2D grid
        interpolate: Whether to interpolate missing values (cubic)
        
    Returns:
        grid_data: (time_steps, grid_size, grid_size) 2D EEG grid
    """
    time_steps = eeg_data.shape[0]
    grid_data = np.zeros((time_steps, grid_size, grid_size), dtype=np.float32)
    
    # Place electrode values on grid
    for t in range(time_steps):
        for ch_idx, (x, y) in enumerate(grid_pos):
            x_int = int(np.clip(np.round(x), 0, grid_size-1))
            y_int = int(np.clip(np.round(y), 0, grid_size-1))
            grid_data[t, y_int, x_int] = eeg_data[t, ch_idx]
    
    # Optional: interpolate missing values
    if interpolate:
        try:
            from scipy.interpolate import griddata
            for t in range(time_steps):
                # Find electrodes with values
                mask = grid_data[t] != 0
                if mask.sum() > 3:  # Need at least 3 points for interpolation
                    points = np.argwhere(mask)
                    values = grid_data[t][mask]
                    grid_coords = np.mgrid[0:grid_size, 0:grid_size].T.reshape(-1, 2)
                    grid_data[t] = griddata(
                        points, values, grid_coords, method='cubic'
                    ).reshape(grid_size, grid_size)
                    # Fill remaining NaN with nearest neighbor
                    mask_nan = np.isnan(grid_data[t])
                    if mask_nan.any():
                        grid_data[t][mask_nan] = griddata(
                            points, values, grid_coords, method='nearest'
                        ).reshape(grid_size, grid_size)[mask_nan]
        except ImportError:
            print("Warning: scipy.interpolate not available, skipping interpolation")
    
    return grid_data


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


class CNN2DSpatialFilter(nn.Module):
    """2D CNN Spatial Filter using electrode topology from MNE."""
    
    def __init__(self, num_sensor=75, num_hidden=500, activation='GELU', 
                 grid_size=64, electrode_file='anatomy/electrode_75.mat',
                 conv_layers=3, dropout=0.15):
        """
        Args:
            num_sensor: Number of electrodes (75)
            num_hidden: Output feature dimension (500)
            activation: Activation function name
            grid_size: Size of 2D electrode grid (default 64x64)
            electrode_file: Path to electrode_75.mat
            conv_layers: Number of 2D convolutional layers
            dropout: Dropout rate
        """
        super(CNN2DSpatialFilter, self).__init__()
        
        self.num_sensor = num_sensor
        self.num_hidden = num_hidden
        self.grid_size = grid_size
        
        # Load electrode positions for grid creation
        try:
            self.grid_pos, _ = create_electrode_grid_mapping(electrode_file, grid_size)
        except Exception as e:
            print(f"Warning: Could not load electrode topology ({e}). Using default grid.")
            self.grid_pos = None
        
        # Input: 1 channel (EEG values on grid)
        in_channels = 1
        
        # Build convolutional layers with increasing filters
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        filter_sizes = [32, 64, 128][:conv_layers]
        for i in range(conv_layers):
            out_channels = filter_sizes[i]
            
            self.conv_layers.append(nn.Conv2d(
                in_channels, out_channels, kernel_size=3, 
                padding=1, stride=1, bias=True
            ))
            self.norms.append(nn.BatchNorm2d(out_channels))
            self.dropouts.append(nn.Dropout2d(dropout * 0.5))
            
            in_channels = out_channels
        
        # Adaptive pooling to get fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers to output dimension
        final_conv_features = filter_sizes[-1] * 4 * 4
        self.fc1 = nn.Linear(final_conv_features, num_hidden)
        self.fc_norm = nn.LayerNorm(num_hidden)
        self.fc_dropout = nn.Dropout(dropout)
        
        self.value = nn.Linear(num_hidden, num_hidden)
        
        self.activation = nn.__dict__[activation]() if activation in nn.__dict__ else nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, time_steps, 75)
                - Could be (batch_size, time_steps, grid_size, grid_size) 
                  if pre-converted by data loader
        
        Returns:
            out: Dictionary with 'value' and 'value_activation' keys
                Output shape: (batch_size, time_steps, num_hidden)
        """
        batch_size, time_steps = x.shape[0], x.shape[1]
        
        # Check if input is already 2D grid or needs conversion
        if x.ndim == 3:  # (batch, time_steps, 75)
            # Convert to 2D grid: (batch*time_steps, 1, grid_size, grid_size)
            x_flat = x.view(-1, 75)  # (batch*time_steps, 75)
            
            if self.grid_pos is not None:
                grid_data = np.zeros((x_flat.shape[0], self.grid_size, self.grid_size), 
                                    dtype=np.float32)
                x_np = x_flat.detach().cpu().numpy()
                for i in range(x_flat.shape[0]):
                    for ch_idx, (gx, gy) in enumerate(self.grid_pos):
                        gx_int = int(np.clip(np.round(gx), 0, self.grid_size-1))
                        gy_int = int(np.clip(np.round(gy), 0, self.grid_size-1))
                        grid_data[i, gy_int, gx_int] = x_np[i, ch_idx]
                
                x_grid = torch.from_numpy(grid_data).to(x.device)
            else:
                # Fallback: Pad 75 channels to 64×64 grid with zeros
                # Create a 64×64 grid and fill first 75 positions (row-major)
                grid_data = np.zeros((x_flat.shape[0], self.grid_size, self.grid_size), 
                                    dtype=np.float32)
                x_np = x_flat.detach().cpu().numpy()
                for i in range(x_flat.shape[0]):
                    # Fill grid row-by-row (channels 0-63 in row 0, 64-74 in row 1, etc.)
                    grid_1d = grid_data[i].flatten()
                    grid_1d[:75] = x_np[i, :75]
                    grid_data[i] = grid_1d.reshape(self.grid_size, self.grid_size)
                
                x_grid = torch.from_numpy(grid_data).to(x.device)
            
            x = x_grid.unsqueeze(1)  # (batch*time_steps, 1, grid_size, grid_size)
        else:  # Assume (batch*time_steps, 1, grid_size, grid_size)
            x = x.view(-1, 1, self.grid_size, self.grid_size)
        
        # Apply convolutional layers
        for conv, norm, dropout in zip(self.conv_layers, self.norms, self.dropouts):
            residual = x if x.shape == conv(x).shape[:] else None
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = dropout(x)
            # Residual connection if shapes match
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch*time_steps, channels, 4, 4)
        
        # Flatten to features
        x = x.view(x.shape[0], -1)  # (batch*time_steps, channels*16)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_norm(x)
        x = self.activation(x)
        x = self.fc_dropout(x)
        
        # Output projection
        value = self.value(x)
        value_activation = self.activation(value)
        
        # Reshape back to (batch, time_steps, num_hidden)
        value = value.view(batch_size, time_steps, -1)
        value_activation = value_activation.view(batch_size, time_steps, -1)
        
        out = dict()
        out['value'] = value
        out['value_activation'] = value_activation
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
