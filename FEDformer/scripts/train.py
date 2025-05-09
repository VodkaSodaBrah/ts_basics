import os
import sys
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Ensure the FEDformer package directory is importable
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fedformer.models.FEDformer import Model as FEDformer

class Configs:
    """
    Wrapper for configuration values, matching FEDformer expected attributes.
    """
    def __init__(self, cfg):
        # defaults adapted from repository's __main__ block
        self.version = cfg.get('version', 'Wavelets')
        self.mode_select = cfg.get('mode_select', 'random')
        self.modes = cfg.get('modes', 32)
        self.seq_len = cfg['model']['input_len']
        self.label_len = cfg['model']['input_len'] // 2
        self.pred_len = cfg['model']['pred_len']
        self.output_attention = cfg.get('output_attention', False)
        self.moving_avg = cfg.get('moving_avg', [12, 24])
        self.L = cfg.get('L', 1)
        self.base = cfg.get('base', 'legendre')
        self.cross_activation = cfg.get('cross_activation', 'tanh')
        self.enc_in = cfg.get('enc_in', 1)
        self.dec_in = cfg.get('dec_in', 1)
        self.d_model = cfg.get('d_model', 512)
        self.embed = cfg.get('embed', 'timeF')
        self.freq = cfg.get('freq', 'h')
        self.dropout = cfg.get('dropout', 0.05)
        self.n_heads = cfg.get('n_heads', 8)
        self.d_ff = cfg.get('d_ff', 2048)
        self.e_layers = cfg.get('e_layers', 2)
        self.d_layers = cfg.get('d_layers', 1)
        self.c_out = cfg.get('c_out', 1)
        self.activation = cfg.get('activation', 'gelu')
        self.wavelet = cfg.get('wavelet', 0)
        self.factor = cfg.get('factor', 1)

def load_yaml_config():
    path = os.path.join(project_root, "configs", "bnb_config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device ➜ {device}")

    cfg_dict = load_yaml_config()
    configs = Configs(cfg_dict)

    # Load preprocessed data (from ts_basics/data/)
    data_path = os.path.abspath(os.path.join(project_root, "..", cfg_dict["data"]["tensor_path"]))
    X, Y = torch.load(data_path)

    ds = TensorDataset(X, Y)
    loader = DataLoader(
        ds,
        batch_size=cfg_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    # Initialize model
    model = FEDformer(configs).to(device)
    # Determine time-feature dimension for dummy embeddings
    d_mark = model.enc_embedding.temporal_embedding.embed.weight.shape[1] if hasattr(model.enc_embedding.temporal_embedding, 'embed') and hasattr(model.enc_embedding.temporal_embedding.embed, 'weight') else model.enc_embedding.temporal_embedding.embed.out_features
    # Compute decoder sequence length for creating dummy time features
    dec_seq_len = configs.label_len + configs.pred_len
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_dict["training"]["lr"]))
    criterion = torch.nn.MSELoss()

    for epoch in range(cfg_dict["training"]["epochs"]):
        total_loss = 0.0
        # Progress bar for batches
        for bx, by in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg_dict['training']['epochs']}", unit="batch"):
            bx, by = bx.to(device), by.to(device)
            # Create dummy time-feature tensors matching required dimension
            zeros_enc = torch.zeros(bx.size(0), bx.size(1), d_mark, device=device)
            # Use full decoder length (label_len + pred_len) for time features
            zeros_dec = torch.zeros(bx.size(0), dec_seq_len, d_mark, device=device)
            # Forward pass: enc_x, enc_time_feat, dec_x, dec_time_feat
            out = model(bx, zeros_enc, by, zeros_dec)
            loss = criterion(out, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1:03d} ─ avg loss {avg:.6f}")

if __name__ == "__main__":
    main()