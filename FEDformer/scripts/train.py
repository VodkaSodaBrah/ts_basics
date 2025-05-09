import os
import sys
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Determine if we should use mixed precision (only on CUDA)
use_amp = torch.cuda.is_available()

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

    # Load config
    cfg_dict = load_yaml_config()
    configs = Configs(cfg_dict)

    # Load preprocessed data
    data_path = os.path.abspath(os.path.join(project_root, "..", cfg_dict["data"]["tensor_path"]))
    X, Y = torch.load(data_path)

    # Dataset / DataLoader with pin_memory
    ds = TensorDataset(X, Y)
    loader = DataLoader(
        ds,
        batch_size=cfg_dict["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model
    model = FEDformer(configs).to(device)
    # Only compile on CUDA to avoid MPS Inductor issues
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Prepare mixed-precision scaler (no-op if not using AMP)
    scaler = GradScaler() if use_amp else None

    # Compute dummy embedding dims
    d_mark = (
        model.enc_embedding.temporal_embedding.embed.weight.shape[1]
        if hasattr(model.enc_embedding.temporal_embedding, 'embed') and
           hasattr(model.enc_embedding.temporal_embedding.embed, 'weight')
        else model.enc_embedding.temporal_embedding.embed.out_features
    )
    dec_seq_len = configs.label_len + configs.pred_len

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_dict["training"]["lr"]))
    criterion = torch.nn.MSELoss()

    # Training loop with progress bar
    for epoch in range(cfg_dict["training"]["epochs"]):
        total_loss = 0.0
        for bx, by in tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{cfg_dict['training']['epochs']}",
            unit="batch",
            miniters=1,
            unit_scale=True
        ):
            # Move to device (non_blocking for pinned memory)
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            # Dummy time features
            zeros_enc = torch.zeros(bx.size(0), bx.size(1), d_mark, device=device)
            zeros_dec = torch.zeros(bx.size(0), dec_seq_len, d_mark, device=device)

            optimizer.zero_grad()
            if use_amp:
                # Mixed precision path
                with autocast(device_type='cuda'):
                    out = model(bx, zeros_enc, by, zeros_dec)
                    loss = criterion(out, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 path
                out = model(bx, zeros_enc, by, zeros_dec)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1:03d} ─ avg loss {avg:.6f}")

if __name__ == "__main__":
    main()