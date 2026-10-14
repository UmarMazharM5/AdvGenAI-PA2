# PA2 evaluation summary

https://github.com/UmarMazharM5/AdvGenAI-PA2

| Comparison | N | Wins | Ties | Losses | Win% excl. ties | Win% ties-half |
| --- | --- | --- | --- | --- | --- | --- |
| HH DPO vs SFT | 300 | 154 | 38 | 108 | 58.8% | 57.7% |
| UltraFeedback DPO vs SFT | 300 | 103 | 49 | 148 | 41.0% | 42.5% |
| Persona LoRA vs SmolLM2-Instruct | 20 | 3 | 14 | 3 | 50.0% | 50.0% |

Win rate excluding ties = W/(W+L). Ties-half = (W+0.5T)/(W+T+L).

Judge: google/gemini-3.6-flash through Kaggle-supported model access. All 620 primary decisions are resolved; unknown and unresolved counts are zero in every track. Saved model responses use identical greedy decoding within each comparison: maximum 256 new tokens, repetition penalty 1.1 and no-repeat n-gram size 4. Model names are hidden from the judge, with A/B positions randomized using seed 42. Judge batches contain up to five pairs, with at least five seconds between attempts, up to three attempts, checkpoints and resume.

HH DPO vs SFT: 5 successful pairs retried, 0 recovered individually; UltraFeedback DPO vs SFT: 30 successful pairs retried, 10 recovered individually; Persona LoRA vs SmolLM2-Instruct: 0 successful pairs retried, 0 recovered individually.

HH DPO wins more often overall, but the category analysis and manually reviewed outputs show remaining unsafe assistance and weaker self-harm support. UF DPO loses more often than SFT. Persona results are balanced, with little target style: both models average 1.00/5 for style and 1.95/5 for usefulness; safety averages are 4.35/5 tuned versus 4.30/5 baseline. These are observed results, not evidence that every DPO track improved.

Two UF training prompts overlap the prescribed AlpacaEval set. Excluding them leaves N=298, W/T/L=103/49/146, win rate excluding ties 41.4% and ties-half 42.8%. This supplementary analysis preserves the conclusion and does not replace the required fixed 300-prompt score.

The primary judgments were produced locally through the supported Kaggle proxy. A saved private Kaggle utility benchmark then verified/resumed all 620 and made 15 separate fresh Gemini 3.6 audit judgments; these did not replace the primary scores. Historical authentication/parsing failures are logged separately and are not model losses. Controlled DPO/persona sanity responses were manually reviewed during the final audit; the timing deviation is disclosed in the main report.

Repository paths: results/generations/ for fixed responses and blind mappings; results/judgments/ for primary decisions, summaries and failure logs; results/audit/ for data/category checks; results/utility/ for the native utility audit.
