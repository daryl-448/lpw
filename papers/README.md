# Paper digests

These are condensed digests of the source papers, for replication and for synthesizing new ideas. The source PDFs are local only: they are git-ignored and will be deleted.

Each digest keeps equations and table numbers verbatim from the paper. It also flags any errors or inconsistencies in the source paper under "inconsistencies" or "gotchas".

| File | Paper | One-line gist |
|---|---|---|
| [banksformer.md](banksformer.md) | Nickerson et al., ECML PKDD 2022 | Transformer-decoder for bank transaction sequences. Uses a product-of-cyclic-date-marginals **date mechanism** and per-field conditional generation (tcode → date → amount). |
| [bankgan.md](bankgan.md) | Mehri et al., Canadian AI 2024 | Hybrid model: a transformer generates the Tcode sequence, then a CTGAN generates the other fields per row, using the Banksformer date mechanism. Compares clock, RBF and one-hot date encodings and runs DP-SGD. |
| [finesse.md](finesse.md) | Farnan et al. (Capital One), arXiv 2026 | Credit-card ABM with latent customer states and 4 coupled event streams (transactions, payments, account state, policy). Benchmark tasks: fraud, missed payment, balance forecasting and next event. |
| [eventflow.md](eventflow.md) | Kerrigan, Nelson & Smyth, AISTATS 2026 | Non-autoregressive TPP generator: first predict the count `n`, then move `n` sorted reference times jointly with flow matching. Uses balanced couplings and an OT-regularised count head. Best multi-step forecasts on 7 unmarked benchmarks. Event times only, no marks. |
| [intensity-free-tpp.md](intensity-free-tpp.md) | Shchur, Biloš & Günnemann, ICLR 2020 | IFTPP / LogNormMix: models the next inter-event time density directly (RNN + log-normal mixture head) instead of the intensity. Closed-form likelihood, mean and reparametrised sampling. Also covers flow variants, marks, metadata, sequence embeddings and imputation. |
| [transxion.md](transxion.md) | Chen et al., KDD 2026 | AML graph benchmark: a profile-rich ABM backbone plus GRPO-trained LLM edits of real illicit subgraphs, embedded under constraints. Detectors score much lower on it than on AMLSim, AMLWorld or SAML-D. |

Lineage: Banksformer → BankGAN (same MUN and Verafin group, which reuses the date mechanism and the Czech/Berka evaluation). FINESSE and TransXion are independent simulator and benchmark lines. IFTPP → EventFlow is the TPP line: EventFlow uses IFTPP as a baseline and replaces its one-step autoregressive density with whole-sequence flow matching.
