# Regular Transformers

 • Order‑agnostic attention: By default, self‑attention treats every position equally and needs positional embeddings to recover “which time‑step is which.”
 • Weak temporal inductive bias: Positional encodings help, but they don’t teach the model how sequences really flow. As the lookback window lengthens, vanilla Transformers often lose accuracy.
• Quadratic cost: Attention scales as O(L²), so very long histories become too slow and memory‑hungry.

1. Bringing RNNs back in pyramid form (PRE)
1. Multiscale convolution (“bottom‑up”)
 • The model starts with the raw, single‑variable sequence.
 • A stack of 1D convolutions with increasing kernel-and‑stride sizes compresses it into progressively shorter “views”—e.g. 24‑step, 48‑step, 72‑step windows—each capturing a different cycle length (daily, bi‑daily, etc.).
1. Upsampling “top‑down” & fusion
 • The model upsamples each coarse feature map back to the original length and merges it with the finer map, so large‑cycle patterns guide smaller‑cycle details.
1. Scale‑wise GRUs
 • Each of those multiscale streams is run separately through a small GRU (or other RNN), whose hidden state learns order‑sensitive dynamics at that timescale.
 • All final hidden states are weighted (via a learned softmax) and summed to produce a single embedding vector for the variable.

This results in a compact, order-aware embedding per variable that fuses cycles from minutes up to days, at a linear cost.

1. Feeding PRE into a Transformer encoder
 • Positional embeddings are omitted—PRE has already baked in sequence order.
 • Multi‑head self‑attention then learns cross‑variable interactions in embedding space, not raw time‑domain, so the Transformer’s strength at modeling multivariate dependencies is preserved.
 • A simple linear projection of each variable’s final encoder output gives the multi‑step forecast.

Why?
 • Stronger temporal inductive bias: RNNs naturally track sequence order; multiscale convolution gives explicit cycle‑level structure.
 • Longer lookbacks become beneficial: PRE compresses long inputs into manageable summaries, so hundreds of steps can be fed without vanishing gradients.
 • Linear‑plus‑quadratic hybrid: PRE runs in O(L), the Transformer in O(D²) where D≪L—so total scales almost linearly with history length.

1. Empirical highlights
 • PRE boosts any Transformer: Simply swapping in PRE for positional encodings cut errors by 30–50% on vanilla, Reformer, Informer, etc.
 • PRformer itself wins 25 out of 32 MSE benchmarks (and 28/32 MAE) across 7 real‑world datasets (electricity, traffic, weather, solar, ETT).
 • Long‑window gains: Unlike pure Transformers, PRformer’s error keeps dropping as the lookback extends from 48→720 steps.
 • Faster and smaller: On Electricity/Traffic/Weather, PRformer trains in about half the time and uses 30–90% of the memory of leading patch‑based models.

PRformer marries a pyramidal RNN front end (for order‑aware, multiscale summarization) to a standard Transformer back end (for cross‑series reasoning). That combo unlocks much longer lookbacks, stronger temporal modeling, and state‑of‑the‑art accuracy on multivariate forecasting—all while keeping compute nearly linear in history length.

## Transformer Details

- Order‑agnostic attention: By default, self‑attention treats every position equally and needs positional embeddings to recover “which time‑step is which.”
- Weak temporal inductive bias: Positional encodings help, but they don’t teach the model how sequences really flow. As the lookback window lengthens, vanilla Transformers often lose accuracy.
- Quadratic cost: Attention scales as O(L²), so very long histories become too slow and memory‑hungry.

## 2. Bringing RNNs Back in Pyramid Form (PRE)

### 2.1 Multiscale Convolution (“bottom‑up”)

- The model starts with the raw, single‑variable sequence.
- A stack of 1D convolutions with increasing kernel-and‑stride sizes compresses it into progressively shorter “views”—e.g. 24‑step, 48‑step, 72‑step windows—each capturing a different cycle length (daily, bi‑daily, etc.).

### 2.2 Upsampling (“top‑down”) & Fusion

- The model upsamples each coarse feature map back to the original length and merges it with the finer map, so large‑cycle patterns guide smaller‑cycle details.

### 2.3 Scale‑wise GRUs

- Each of those multiscale streams is run separately through a small GRU (or other RNN), whose hidden state learns order‑sensitive dynamics at that timescale.
- All final hidden states are weighted (via a learned softmax) and summed to produce a single embedding vector for the variable.

This results in a compact, order-aware embedding per variable that fuses cycles from minutes up to days, at a linear cost.

## 3. Feeding PRE into a Transformer Encoder

- Positional embeddings are omitted—PRE has already baked in sequence order.
- Multi‑head self‑attention then learns cross‑variable interactions in embedding space, not raw time‑domain, so the Transformer’s strength at modeling multivariate dependencies is preserved.
- A simple linear projection of each variable’s final encoder output gives the multi‑step forecast.

## 4. Why This Helps

- Stronger temporal inductive bias: RNNs naturally track sequence order; multiscale convolution gives explicit cycle‑level structure.
- Longer lookbacks become beneficial: PRE compresses long inputs into manageable summaries, so hundreds of steps can be fed without vanishing gradients.
- Linear‑plus‑quadratic hybrid: PRE runs in O(L), the Transformer in O(D²) where D≪L—so total scales almost linearly with history length.

## 5. Empirical Highlights

- PRE boosts any Transformer: Simply swapping in PRE for positional encodings cut errors by 30–50% on vanilla, Reformer, Informer, etc.
- PRformer itself wins 25 out of 32 MSE benchmarks (and 28/32 MAE) across 7 real‑world datasets (electricity, traffic, weather, solar, ETT).
- Long‑window gains: Unlike pure Transformers, PRformer’s error keeps dropping as the lookback extends from 48→720 steps.
- Faster and smaller: On Electricity/Traffic/Weather, PRformer trains in about half the time and uses 30–90% of the memory of leading patch‑based models.

PRformer marries a pyramidal RNN front end (for order‑aware, multiscale summarization) to a standard Transformer back end (for cross‑series reasoning). That combo unlocks much longer lookbacks, stronger temporal modeling, and state‑of‑the‑art accuracy on multivariate forecasting—all while keeping compute nearly linear in history length.
