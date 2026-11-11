# DNN Programming Assignment-2

Variant: seeds **11, 22, 33** | hidden size **64** | student-ID last digit **9** (odd -> LSTM forget-gate bias 0 vs 1 ablation at T=200).

## Contents
| Path | Description |
|---|---|
| `problem1/problem1.ipynb` | RNN / LSTM / GRU: HAR classification, adding problem, gradient analysis, hidden-size sweep, ablation |
| `problem2/problem2.ipynb` | Transformer from scratch: HAR encoder (Task A), Tiny Shakespeare causal LM (Task B), ablations |
| `problem2/generated_samples.txt` | 500 characters at temperature 0.2 / 0.8 / 1.5 (top-k 20) |
| `problem1/figures/`, `problem2/figures/` | All plots produced by the notebooks |
| `results.csv` | Results table (columns: problem, model, seed, seq_len, test_acc, macro_f1, test_mse, params, sec_per_epoch, val_loss_nats, bits_per_char, notes) |
| `report.pdf` | Analysis and assumptions |

## How to run
1. Python 3.10+, PyTorch, numpy, pandas, matplotlib, scikit-learn. A GPU is used if available (results here: CUDA).
2. In each notebook set `QUICK_RUN = False`, then *Restart and Run All*. Data (UCI HAR, Tiny Shakespeare) is downloaded automatically into `data/`.
3. Run `problem1.ipynb` first, then `problem2.ipynb` (the P2 comparison cell reads the P1 rows from `../results.csv`). Each notebook only replaces its own rows (`P1*` / `P2*`), so re-running is safe.
4. Relative paths only; the notebooks work from `problem1/` or `problem2/`.

## Notes on `results.csv`
The notebooks append **one row per run** (per seed). The `results.csv` delivered here is a **summary version**: for multi-seed experiments the row has `seed = mean` with the mean over seeds 11, 22, 33 and the standard deviation in `notes`; Task B rows are single runs (seed 11). It was assembled from the printed tables in the notebook outputs, so values are rounded to the printed precision. For the per-seed file, re-run the notebooks and use the `results.csv` they write.

## Assumptions
- Problem 1: same budget for RNN/LSTM/GRU (Adam lr 1e-3, batch 128, clip 1.0, same epochs); best-validation epoch is evaluated on test; HAR validation = 4 training subjects chosen per seed; normalisation from the training part only; `nn.*` layers for training, from-scratch cells for tests and the gradient plot.
- Problem 2: instructor's 3-token example not available -> hand-checkable example used; Task A patch 8, 2 pre-LN layers, AdamW 5e-4; Task B AdamW 1e-3, 100-step warm-up + cosine decay, validation on the whole last 10% in non-overlapping 128-character windows. `nn.MultiheadAttention` / `nn.Transformer*` are not used (`F.scaled_dot_product_attention` only inside a unit test).
- Language model: d=128, 4 layers, 4 heads (GPU run).

## Headline results
- HAR (hidden 64): RNN 0.8422, LSTM 0.8973, GRU 0.8896 test accuracy.
- Adding problem: GRU solves T=50-400 (3/3); LSTM fails from T=200; RNN from T=100.
- Transformer HAR: 0.9313 (no PE), 0.9104 (sinusoidal), 0.9092 (learned).
- Char LM: 1.6753 nats = 2.417 bits/char validation.

## GitHub repository
[Assignment_Bits / 2026AJ05499_DNN_Assignment (branch `DNN_Assignment`)](https://github.com/officialinmailbox-art/Assignment_Bits/tree/DNN_Assignment/2026AJ05499_DNN_Assignment)
