{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 Fit GARCH(1,1) Directly (No ARIMA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from arch import arch_model\n",
    "\n",
    "# Directly fit a GARCH(1,1) to captures both mean=0 and volatility dynamics\n",
    "garch = arch_model(returns, mean=\"Zero\", vol=\"Garch\", p=1, q=1)\n",
    "res = garch.fit(disp=\"off\")\n",
    "\n",
    "# Print summary\n",
    "print(res.summary())"
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
