# Advanced Generative AI PA2

Original SFT and DPO implementation for the assignment in PA2.pdf.
Starter tests and fixed data are from https://github.com/ahmad903501/small-model-alignment-starter.

## Current status

- Core response-only batching, causal scoring, SFT loss, and DPO loss implemented.
- Official public tests and additional truncation/numerical tests pass locally.
- Original mixed-precision training runner saves best/final checkpoints and JSONL metrics.
- Data preparation, GPU preflight and executed Kaggle notebook sources preserved.
- All four training tracks and required generations completed on Kaggle.
- All 620 pairwise judgments resolved using google/gemini-3.6-flash.
- HH DPO vs SFT: 154 wins, 38 ties, 108 losses; 58.8% excluding ties.
- UltraFeedback DPO vs SFT: 103 wins, 49 ties, 148 losses; 41.0% excluding ties.
- Persona vs SmolLM2-Instruct: 3 wins, 14 ties, 3 losses; limited style change.
- Four training plots and the final eight-page report generated:
  [PA2_report.pdf](report/PA2_report.pdf).
- Repository: https://github.com/UmarMazharM5/AdvGenAI-PA2 (private; grant course
  staff access before submitting the repository URL).
- See RUN_STATUS.md for recovery history. The packaged project stores run evidence
  under results/training, results/generations, results/judgments and results/plots.

## Kaggle credentials

Copy `.env.example` to `.env` and fill either `KAGGLE_API_TOKEN`, or the legacy
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Never put credentials in notebook cells.
The local wrapper loads `.env` without printing its contents:

```powershell
python scripts/kaggle_cli.py kernels list --mine
```

## Notebooks and tests

```powershell
python -m pytest -q tests
python scripts/build_notebooks.py
```

`notebooks/00_preflight.ipynb` verifies two T4 GPUs, dependencies, and offline tests.
`notebooks/01_sft.ipynb` prepares data and runs full SFT. Both embed an allowlist
of project files, so execution does not depend on a published GitHub repository.
Use internet access and GPU T4 x2. Save versions after each completed stage.

For local orchestration/report tools, install `requirements-local.txt`.
The packaged evidence can be independently verified with
`python scripts/verify_submission.py`. The report can be regenerated with
`python scripts/build_report.py --repo-url <your GitHub repository URL>`.

SFT reduction is the mean negative log probability over scored response tokens.
The fixed template is `### Instruction:\n{instruction}\n\n### Response:\n`.
Prompts and responses are tokenized separately; EOS is appended to the response.
SFT extracts the first user/assistant pair; DPO excludes multi-turn pairs.
Training data excludes prompts exceeding 256 tokens and uses sequence length 512.
Validation prompts are disjoint from training prompts. Dataset fingerprints and
subset hashes are saved in `outputs/data/manifest_<track>.json`.

Main generation uses greedy decoding, maximum 256 new tokens, repetition penalty
1.1, and no-repeat n-gram size 4 after a plain greedy sanity response looped.
Compared models always use identical settings. Persona generation uses SmolLM2's
chat template and a separate fixed 20-prompt set covering cooking, coding,
general help, and safety. Persona is compared with untuned SmolLM2-Instruct.

The evaluator reads saved JSONL only, blinds model names, judges in batches of
five with at least five seconds between attempts, allows three attempts, and
resumes completed pairs. It reports wins/losses/ties/unknown/failures with explicit
denominators. Kaggle-supported proxy credentials stay in ignored `.env.benchmarks`.
The proxy HTTP route follows Kaggle Benchmarks' official ModelProxy client:
https://github.com/Kaggle/kaggle-benchmarks/blob/master/kaggle_benchmarks/kaggle/model_proxy.py
The original 620 decisions ran locally through this supported interface. A saved
private utility benchmark then resumed/verified all 620 and made fifteen fresh
blinded audit judgments with Gemini 3.6; those separate decisions do not replace
the primary scores. Kaggle also automatically validated task creation with
Gemini 3.8; those outputs remain separate. The saved utility task is
https://www.kaggle.com/benchmarks/tasks/umarmazhar2025/pa2-saved-pairwise-evaluation/2.
Its source is `notebooks/05_evaluation_utility.ipynb`. Official task-creation docs:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md.

Training follows the guide settings: FP32 trainable weights with FP16 autocast,
gradient scaling, clipping at 1.0, AdamW, SFT cosine decay and 3% warmup; DPO
uses fixed learning rate and a frozen FP16 reference on the second GPU.
Gradient checkpointing reduces memory use. All four actual runs completed.

The final audit verified the exact saved subset counts, hashes, prompt token
limits and train/validation separation in a private CPU Kaggle notebook. Two UF
training prompts also appear in the prescribed AlpacaEval sample. The fixed
300-prompt result remains 41.0% decisive wins; excluding those two gives 41.4%
over 298 prompts. The report includes category outcomes and visible failures;
relative HH wins do not establish absolute safety, and persona style was weak.

Controlled DPO/persona sanity cells originally ran automatically before full
generation; manual review happened during the final audit. Updated generation
notebooks require `SANITY_REVIEWED=True` after inspection. This timing deviation
is disclosed in the report. SFT gradient accumulation averages microbatch token
means, while validation loss is weighted by scored response-token counts.

The assignment requires independent implementation and citations. Review and
understand generated code before submission and follow your course's AI-use policy.
