# PA2 execution status

## Current status: experiments complete

All four training tracks and all required generations completed. All 620 blinded
judgments are resolved with zero unknown or unresolved results. Judge:
google/gemini-3.6-flash through Kaggle-supported model access.

| Comparison | Wins | Ties | Losses | Win rate excluding ties |
| --- | ---: | ---: | ---: | ---: |
| HH DPO vs SFT | 154 | 38 | 108 | 58.8% |
| UltraFeedback DPO vs SFT | 103 | 49 | 148 | 41.0% |
| Persona LoRA vs SmolLM2-Instruct | 3 | 14 | 3 | 50.0% |

The eight-page report and four training plots are generated. The submission
archive and project folder are under `submission/`. Verification recomputed all
620 outcomes, checked matching generations/decoding settings, verified artifact
hashes, and found no credentials in the package. Nineteen offline tests pass,
including independent non-uniform causal-scoring and gradient checks.

A saved private Kaggle utility benchmark now verifies/resumes all 620 primary
decisions and separately judges fifteen fixed saved pairs with Gemini 3.6.
The original scores remain unchanged. Task creation also ran an automatic
Gemini 3.8 validation; those outputs remain separate. Utility task:
https://www.kaggle.com/benchmarks/tasks/umarmazhar2025/pa2-saved-pairwise-evaluation/2.
A private CPU audit verified all six exact saved training/validation subsets,
including hashes, counts, prompt limits and separation. Two UF training prompts
match AlpacaEval; the supplementary 298-prompt decisive win rate is 41.4%.
The report documents uneven HH safety outcomes, weak persona style, and the
controlled-sanity review timing deviation. Updated generation notebooks gate
full generation on manual sanity review.

GitHub login was renewed successfully as UmarMazharM5. The project repository is
https://github.com/UmarMazharM5/AdvGenAI-PA2 (private; grant course staff access).
The final PDF includes this real repository URL. Queue/monitor processes have
completed their work. Earlier authentication failures below are historical.

## Earlier execution and recovery notes (historical snapshots)

Kaggle account: umarmazhar2025. All notebooks are private.

- Preflight: completed successfully. Two Tesla T4 GPUs, 15,360 MiB each.
  PyTorch 2.11.0+cu128, Transformers 4.57.6, Datasets 4.8.5.
  All 11 tests passed on Kaggle.
  https://www.kaggle.com/code/umarmazhar2025/pa2-preflight
- SFT notebook version 1: failed during data preparation with a Hugging Face
  HTTP 429 error while loading HH-RLHF. No training updates were reached.
- SFT notebook version 2: submitted after isolating SFT data preparation and
  adding bounded retries with delays for transient download errors.
  Live logs confirm 30,000 train and 1,000 validation rows saved. Two-step pilot
  completed with finite losses and no skipped optimizer updates. Pilot train
  losses: 1.4544, 1.3348; pilot validation loss: 1.5145 before, 1.5232 after.
  Full SFT completed all 938 steps. Initial validation loss 1.4150; final and
  best validation loss 1.3875. Zero skipped optimizer updates. The selected best
  checkpoint is the final step 938 checkpoint. Metrics and checkpoint metadata
  were retrieved into `outputs/sft_saved/pa2/outputs/sft`.
  It prepares SFT data and fixed AlpacaEval prompts, runs a two-step memory/precision pilot,
  then trains SFT on 30,000 examples with 1,000 validation examples.
  https://www.kaggle.com/code/umarmazhar2025/pa2-sft-training
- Saved best/final checkpoint metadata and completed.json are verified.
- A private three-prompt base-versus-SFT generation sanity notebook completed:
  https://www.kaggle.com/code/umarmazhar2025/pa2-sft-sanity
- SFT loss plot exported to `outputs/plots/sft.png` and `outputs/plots/sft.pdf`.

## Preference training and queued evaluation

Recovery update: both full DPO runs completed; best HH reward accuracy is 0.648
(step 469), best UltraFeedback accuracy is 0.706 (step 400), with final UF 0.698.
Each had three AMP-skipped updates and 466 successful updates. Both 300-prompt
generation comparisons completed and were downloaded. Persona version 1 failed
at step zero with a DataParallel device mismatch; version 2 restricted its TRL
process to one GPU and completed 39 updates. The selected persona checkpoint is
step 13 (first tied maximum of 100% reward accuracy on 22 validation pairs).

The first full judge attempts encountered expired model-proxy credentials and
produced zero valid judgments. These are interface failures, not model losses.
Credentials were renewed; the evaluator now refreshes on HTTP 401 and stops
repeated authentication denial. Evaluation resumed on the existing saved pairs.
Historical failure logs remain intact. Report generation waits for all three
evaluation summaries to have zero unresolved failures. GitHub authentication is
still required for a submission-ready repository URL; otherwise the report is
explicitly labeled DRAFT.

- HH DPO: running, 15,000 training pairs / 1,000 validation pairs. Latest inspected
  step 12/469, finite training loss and finite gradient norm.
  https://www.kaggle.com/code/umarmazhar2025/pa2-dpo-hh
- UltraFeedback DPO: running, same sizes and same starting SFT checkpoint.
  Latest inspected step 5/469, finite training loss and finite gradient norm.
  https://www.kaggle.com/code/umarmazhar2025/pa2-dpo-uf
- Both AMP scalers skipped updates at steps 1 and 4 during startup overflow;
  subsequent inspected updates succeeded. Preserve these records in the report.
- Initial reward accuracy is 0.514 (HH) / 0.470 (UF), with near-zero reward margins.
  Unlike mathematically identical policy/reference calculations, FP32 policy
  weights and FP16 reference weights create small scoring differences before any
  update. Initial accuracy must not be interpreted as learned preference ability.
- Plain greedy SFT sanity generation repeated a bear name up to the token limit.
  Main evaluation will use repetition_penalty=1.1 and no_repeat_ngram_size=4,
  identically for all compared models. This decoding deviation will be reported.
- Kaggle-supported google/gemini-3.6-flash judging was verified on three saved
  blinded sanity pairs. These are interface checks, not final benchmark results.
- Kaggle's two-GPU-session account limit deferred persona training. A local
  queue polls every minute, then launches the prepared persona and generation
  notebooks as slots free. It judges only saved pairwise outputs, sequentially.
  Queue state: `outputs/pipeline/status.json`; monitor state:
  `outputs/monitor/hh/status.json` and `outputs/monitor/uf/status.json`.
- The queue records failed training/dependencies and does not restart them
  blindly. This computer must stay on for queued jobs to launch and local judging.
- The queue was verified active after correcting a local active-session counting
  bug; the Kaggle runs were unaffected. State snapshots show both DPO runs active.
- Fourteen local tests pass, including evaluator blinding, winner validation and
  explicit failure denominators. Judge API access succeeded on the real saved
  sanity pairs using the prescribed google/gemini-3.6-flash model.
- GitHub CLI account HeavyFIRE has an expired/invalid stored token. Repository
  publishing will require reauthentication; Kaggle work continues independently.

Monitor:

A hidden local monitor (initial PID 31212) collected live metrics until SFT
completed and has now exited successfully.
Read `outputs/monitor/sft/status.json` for its latest snapshot, `metrics.jsonl`
for observed metrics, and `alerts.jsonl` if an error occurs. It records errors;
it does not automatically restart training. The local monitor requires this
computer to stay running; Kaggle training continues independently.

```powershell
python -X utf8 scripts/kaggle_cli.py kernels status umarmazhar2025/pa2-sft-training
python -X utf8 scripts/kaggle_cli.py kernels logs umarmazhar2025/pa2-sft-training
```

Next: inspect SFT metrics and a few generated responses, then execute each DPO
notebook with the saved SFT notebook output attached. Both use the same SFT best
checkpoint. Persona TRL/LoRA and Kaggle model judging still require implementation
and execution. Final report must use observed results only.
