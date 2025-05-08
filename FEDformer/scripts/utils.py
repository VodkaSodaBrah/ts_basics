import os
import sys
from fedformer.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from fedformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from fedformer.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from fedformer.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from fedformer.layers.SelfAttention_Family import FullAttention, ProbAttention
from fedformer.layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp, series_decomp_multi
)

# ---- add project paths so "fedformer", "utils", etc. import cleanly ----
project_root = os.path.abspath("..")                       # …/FEDformer
fed_root     = os.path.join(project_root, "fedformer")     # …/FEDformer/fedformer
utils_root   = os.path.join(fed_root, "utils")
layers_root  = os.path.join(fed_root, "layers")

# Put lowest‑level dirs first so they shadow any third‑party “utils” module
for p in (utils_root, layers_root, fed_root, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)
# -----------------------------------------------------------------------

from fedformer.utils.masking import LocalMask   # ← import now resolves