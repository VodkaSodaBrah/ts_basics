{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GARCH Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Note on Heavy Tails:**  \n",
    "While GARCH(1,1) with Gaussian residuals captures volatility clustering, its standardized residuals often exhibit “fat tails” (extreme events) that a normal distribution underestimates, which can lead to underestimating risk measures like Value‐at‐Risk (VaR) or Expected Shortfall. citeturn1file2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1 Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 Load Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Risk Measure Context:**  \n",
    "Beyond simple volatility, coherent risk measures—such as Expected Shortfall—are preferred tools in risk management because they satisfy key properties like subadditivity and monotonicity. This helps ensure our models reflect worst‐case losses more accurately. citeturn1file1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# code to compute returns here"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 Fit GARCH(1,1) Directly (No ARIMA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from arch import arch_model\n",
    "\n",
    "# Directly fit a GARCH(1,1) to captures both mean=0 and volatility dynamics\n",
    "garch = arch_model(returns, mean=\"Zero\", vol=\"Garch\", p=1, q=1)\n",
    "res = garch.fit(disp=None)\n",
    "\n",
    "# Print summary\n",
    "print(res.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Non‐Parametric Innovations:**  \n",
    "Rather than assuming Gaussian or t‐distributed innovations, one can fit the standardized residuals with a kernel density (Silverman bandwidth). Then reestimate GARCH parameters by maximizing the likelihood under that density. Repeat until convergence for a highly flexible volatility model. citeturn1file4  \n",
    "**Pseudocode:**  \n",
    "1. Initialize GARCH parameters θ.  \n",
    "2. Loop until convergence:  \n",
    "   - Compute standardized residuals uₜ/σₜ.  \n",
    "   - Fit kernel density f̂ on {uₜ/σₜ}.  \n",
    "   - Update θ by maximizing ∑ₜ [log f̂(uₜ/σₜ) – log σₜ].  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4 Test for ARCH effects"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- **Volatility Clustering (Taylor Effect):** Observations of absolute or squared returns show slowly decaying autocorrelations, indicating that large moves tend to cluster in time. citeturn1file2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Power‐GARCH Note:**  \n",
    "A generalization models σ^δ = γ + α |ε|^δ + β σ^δ. Empirical studies often find δ≈1 (absolute deviations) fits financial returns better than δ=2. citeturn1file2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5 Fit GARCH(1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Empirical Performance:**  \n",
    "Studies on real market data (e.g., IBM daily returns) show that adding non‐parametric innovations can improve backtest metrics like VaR and ES compared to standard GARCH(1,1). citeturn1file5"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
