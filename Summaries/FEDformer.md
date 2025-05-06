# FEDformer Summary

Why standard Transformers struggle with long time series

Vanilla self and cross‑attention treat each time step independently in the time domain, making it hard to capture overall trends and seasonal patterns. They also incur quadratic compute/memory costs as the sequence length grows.

Seasonal‑trend decomposition

FEDformer first splits each series into a seasonal part (high‑frequency fluctuations) and a trend part (long‑term movement) via a Mixture‑of‑Experts Decomposition block. This block uses several average‑pooling filters of different window sizes and learns weights to recombine them, so the model can flexibly extract multiple trend scales rather than relying on a single fixed window.

Frequency‑enhanced attention

Instead of attending in the time domain, FEDformer’s layers use Frequency Enhanced Blocks (FEB) and Frequency Enhanced Attention (FEA) to work in the frequency domain.
  
FEB‑f (FEB with Fourier) replaces self‑attention by:

  1. Projecting inputs to queries/keys/values.

First, take each point in your sequence and convert it into a little summary vector (that’s the “query/key/value” projection). It’s like giving each data point a ticket that says, “Hey, here’s what I’m about.”

  1. Applying the FFT (Discrete Fourier Transform) to get frequency coefficients.

Instead of comparing every ticket to every other ticket one by one, you transform your whole sequence into a set of simple waves (via the Fast Fourier Transform). These waves tell you the overall rhythms in your data—like whether it’s generally rising and falling every day, week, or month.

  1. Randomly selecting a small fixed set (M) of these coefficients.

Real data often only needs a few of those waves to describe its main patterns. So you randomly choose a small fixed number (say, 64 out of hundreds) of these waves to focus on. It’s like tuning in to your favorite radio stations and ignoring all the rest.

  1. Mixing them via a learned kernel and inverse FFT back to time domain.

You learn a tiny mixing recipe (the “kernel”) that says how much weight to give each of those chosen waves. Then you combine them and flip back from the wave world into normal time‑series again (via the inverse FFT). The result is a new sequence that already has built‑in knowledge of the big picture rhythms—so downstream layers can make smarter predictions without having to re‑compare every point with every other point.

FEA‑f replaces cross‑attention similarly, performing the canonical “QKᵀ” dot‑product (with an activation like softmax or tanh) in the reduced M‑mode frequency space before inverting via FFT.

Why FFT matters

  1. A faster shortcut for finding waves
Imagine you had to check every single pair of data points to figure out all the hidden rhythms in a long sequence—that’s what the naïve “DFT” does, and it takes forever as your list grows.

The FFT is a clever trick that discovers those same rhythms by reusing work, so instead of time growing with the square of your list length, it grows only a bit faster than the list itself (roughly “list length × a little bit more”). That makes it practical even for really long sequences.
  
  1. Only paying attention to a few key rhythms

Once you’ve transformed into the “wave world,” you don’t need all of the hundreds or thousands of possible waves—just a small handful that carry the main story (say 64 out of 1,000).

Because you only mix those few waves back into your data, the work you do after the FFT now grows in direct proportion to how long your data is (times that small handful), not the square of its length. In everyday terms, doubling your sequence only roughly doubles the work, rather than quadrupling it.ion.

Random mode selection keeps information

Rather than keeping only low‑frequency modes, FEDformer randomly samples M frequencies (mixing low and high) to capture both smooth trends and abrupt events. A theoretical analysis shows that, if the time series’ Fourier matrix has moderate coherence, selecting O(k²) random modes preserves nearly as much information as the best rank‑k approximation.

Overall complexity and gains

By decoupling sequence length (N) from attention‑matrix dimension (M), FEDformer trains and infers in time/memory O(N) with a single forward pass—matching or beating prior “efficient” Transformers like Autoformer and Informer. It reduces multivariate forecasting error by ∼15% and univariate error by ∼23% across six real‑world benchmarks.

In short, FEDformer marries classic seasonal‑trend decomposition with FFT‑accelerated, frequency‑domain attention to capture both global profiles and fine structures in very long series—while scaling linearly in sequence length.
