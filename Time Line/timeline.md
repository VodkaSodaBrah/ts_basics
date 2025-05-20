<!-- markdownlint-disable MD033 -->
# Intern Project Timeline  

All dates America/Phoenix: GMT-7

---

## 1 . High‑Level Task Schedule

| # | Task & Key Sub‑tasks | Effort (hrs) | **Due‑date** |
|---|----------------------|--------------|--------------|
| **0** | **Coin‑flip ticket‑value problem** – Derive expected value algebraically; Monte‑Carlo simulation (≥100 k trials); notebook | **6** | **Fri 16 May 2025 @ 06:00 AM** |
| 1 | **iTransformer adaptation** – 1. Read paper / repo; design gap list; 2. Build data pipeline & loaders; 3. Run training + hyper‑param sweep; 4. Results notebook | 30 | **Tue 3 Jun 2025** |
| 2 | **“How Does the World Spin” packet** ; – Identify & describe seasonality in Lean Hogs (HE) chart ; – Detrend HE series and plot pattern ; – Explain underlying drivers of the pattern ; – Back‑test at least one seasonal strategy (e.g., tuned RSI) and report results ; – Discuss predictive vs. profitable distinction ; – Answer **all** bolded discussion questions in the doc (pattern, detrending, pigs ↔ Tesla/BTC, BTC safe‑haven, dollar strength, perfect simulator, statistical methods) ;  | 12 | **Tuesday 10 Jun 2025** |
| 3 | **Trading‑strategy proposal** – Data‑source rationale; Feature engineering & model concept; Signal → trade rules, risk, sizing; Slide deck + diagrams | 6 + ongoing| **Fri 13 Jun 2025** |

---

## 2 . Micro‑Plan for Week of 12 May 2025  

### Focus: Coin‑flip assignment

| Date | Focus | Target output by EOD |
|------|-------|----------------------|
| **Tue 13 May** | • Env setup (Python, NumPy/pandas, Jupyter)<br>• Create project board & notebook skeleton | Notebook skeleton with headings **Math Derivation** / **Simulation** |
| **Wed 14 May** | • Derive expected‑value formula (geometric series) (Markdown / LaTeX) | Finished *Mathematical Section* in notebook |
| **Thu 15 May** | • Code Monte‑Carlo sim (≥100 k trials)<br>• Validate mean vs. formula<br>• Polish narrative, add figure | Finalised notebook & exported PDF |
| **Fri 16 May** | **06:00 AM submission** – update repo with `coin_flip_ev.ipynb` + PDF | Send to Mathias |

---

## 2 . Micro‑Plan for Task 1: iTransformer adaptation  

(due Tue 03 Jun 2025)

| Date             | Focus                                             | Target output by EOD                              |
|------------------|---------------------------------------------------|----------------------------------------------------|
| **Mon 19 May**   | • Clone iTransformer repo & set up env<br>• Install dependencies<br>• Sketch delta list outline | Initial `iTransformer_delta_list.md` created       |
| **Wed 21 May**   | • Complete delta list with component gaps<br>• Review FEDformer code sections | Finalized delta list table                        |
| **Thu 22 May**   | • Implement data loader adaptations<br>• Test data pipeline loading | Sample batch prints correctly in notebook          |
| **Mon 26 May**   | • Run initial training run (short epoch)<br>• Hook up logging (W&B or TensorBoard) | Training logs & first loss curve                  |
| **Wed 28 May**   | • Conduct hyper‑parameter sweep<br>• Collect results metrics | Summary table of hyper‑param results               |
| **Thu 29 May**   | • Draft 1‑page summary of findings<br>• Prepare results notebook | `iTransformer_results.ipynb` draft ready          |
| **Tue 03 Jun**   | **Submit Task 1 deliverables** results notebook<br>– summary PDF | Email/PR with all Task 1 artifacts                 |

---

## 3 . Assumptions & Buffers

* **Effort estimates** assume ~20 productive hours/week.  
* Long training jobs (iTransformer) overlap lighter writing tasks (World‑Spin reflection) to keep momentum.  
* Regular Friday hand‑offs provide natural checkpoints for our 1:1 and group meetings.

---

Last updated 17 May 2025 – adjusted due-dates to Fridays
