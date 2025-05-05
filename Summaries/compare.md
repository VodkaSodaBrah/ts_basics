# What each one does first

 • FEDformer analyzes the entire series, decomposing it into “big picture” trends and “seasonal wiggles,” then examines those patterns as repeating waves (via an FFT) instead of point by point.
 • PRformer begins by compressing the long series into a few shorter summaries at different scales (fast, medium, slow) using one‑dimensional convolutions, then applies a recurrent network at each scale to learn temporal dynamics.

How they learn across multiple variables

 • FEDformer still uses attention, but it does it on those selected frequency waves, so it’s comparing global rhythms across different sensors or signals.
 • PRformer hands each variable’s “pyramid + RNN” embedding into a plain Transformer attention layer, so it’s comparing those compact, order‑aware summaries.

Why pick one over the other

 • Choose FEDformer when the data exhibits clear periodic or seasonal cycles (such as electricity demand peaking daily or weekly) and global repeating wave patterns should be emphasized. It’s very efficient even for thousands of steps.
 • Choose PRformer when precise event ordering is critical (for instance, sudden changes or rare spikes) and both short‑term and long‑term rhythms should be captured. Its “pyramid + RNN” front end makes it naturally sensitive to the sequence of events.

How they scale with really long histories

 • Both are designed to handle long lookbacks cheaply (linear in length rather than quadratic), but they do it differently: FEDformer by picking a handful of important frequencies, PRformer by compressing long sequences into much shorter RNN‑friendly streams.

In everyday terms

 • FEDformer resembles tuning a radio to a few key stations that reveal data’s major trends and rhythms.
 • PRformer resembles observing data through time‑lapse lenses (fast, medium, slow), with brief summaries recorded at each scale—then combining those summaries for forecasting.

Both outperform plain Transformers on real‑world forecasting tasks, so it ultimately depends on whether global wave patterns (FEDformer) or sequence‑order and multiscale detail (PRformer) are preferred.
