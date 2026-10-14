"""Generate a report of at most eight pages from completed run evidence."""
import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.tools'))
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-url')
    parser.add_argument('--output', default='report')
    a = parser.parse_args()
    packaged=(ROOT/'results/training').is_dir()
    audit_root=ROOT/'results/audit' if packaged else ROOT/'outputs/audit'
    utility_root=ROOT/'results/utility' if packaged else ROOT/'outputs/utility_saved'
    utility_files=[p for p in utility_root.rglob('utility_completed.json')
                   if '3.6' in json.loads(p.read_text()).get('judge_model','')]
    utility=json.loads(utility_files[-1].read_text()) if utility_files else None
    data_audit_path=audit_root/'data_audit.json'
    data_audit=json.loads(data_audit_path.read_text()) if data_audit_path.exists() else None
    def evaluation_file(track,name):
        return ROOT/f'results/judgments/{track}/{name}' if packaged else ROOT/f'outputs/evaluation/generate-{track}/{name}'
    def generation_file(track,name):
        return ROOT/f'results/generations/{track}/{name}' if packaged else next((ROOT/f'outputs/generated/{track}').rglob(name))
    tracks = ('hh','uf','persona')
    evaluations = {}
    for track in tracks:
        p = evaluation_file(track,'judged.summary.json')
        if not p.exists():
            raise RuntimeError(f'Awaiting completed {track} evaluation')
        evaluations[track] = json.loads(p.read_text())
        if evaluations[track]['failures']:
            raise RuntimeError(f'{track} still has unresolved judge failures; repair before reporting')
    run_paths = {
        'sft':ROOT/'outputs/monitor/sft/artifacts/pa2/outputs/sft',
        'hh':ROOT/'outputs/monitor/hh/artifacts/pa2/outputs/dpo_hh',
        'uf':ROOT/'outputs/monitor/uf/artifacts/pa2/outputs/dpo_uf',
        'persona':ROOT/'outputs/monitor/persona_v2/artifacts/pa2/outputs/persona',
    }
    if packaged:
        run_paths={t:ROOT/f'results/training/{t}' for t in ('sft','hh','uf','persona')}
    metrics = {k:rows(p/'metrics.jsonl') for k,p in run_paths.items() if k!='persona'}
    persona_config = json.loads((run_paths['persona']/'config.json').read_text())
    persona_metrics = json.loads((run_paths['persona']/'metrics.json').read_text())
    validations = {k:[r for r in v if r['phase']=='validation'] for k,v in metrics.items()}
    best = {'sft':min(validations['sft'],key=lambda r:r['loss'])}
    for t in ('hh','uf'):
        best[t] = max(validations[t],key=lambda r:r['reward_accuracy'])
    persona_best_step = int(json.loads((run_paths['persona']/'completed.json').read_text())['best_checkpoint'].rsplit('-',1)[1])
    persona_best = next(r for r in persona_metrics if r.get('step')==persona_best_step and 'eval_loss' in r)
    output = ROOT/a.output
    output.mkdir(parents=True,exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallText',fontSize=8.5,leading=11,spaceAfter=5))
    styles['BodyText'].fontSize=9.3
    styles['BodyText'].leading=12
    story = []
    markdown = []
    def heading(text):
        story.append(Paragraph(escape(text),styles['Heading2']))
        markdown.append('## '+text+'\n')
    def paragraph(text, small=False):
        story.append(Paragraph(escape(text),styles['SmallText' if small else 'BodyText']))
        if not small:
            story.append(Spacer(1,5))
        markdown.append(text+'\n')
    def table(data, widths=None):
        wrapped=[[Paragraph(escape(str(cell)),styles['SmallText']) for cell in row] for row in data]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf5')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.3,colors.lightgrey),
                              ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        story.append(t)
        story.append(Spacer(1,8))
        markdown.extend([' | '.join(str(c) for c in r)+'\n' for r in data])
    def page():
        while story and isinstance(story[-1],Spacer):
            story.pop()
        story.append(PageBreak())
    def figure(name, width, height):
        directory='results/plots' if packaged else 'outputs/plots'
        story.append(Image(str(ROOT/f'{directory}/{name}.png'),width=width*inch,height=height*inch))
        markdown.append(f'![{name}](../{directory}/{name}.png)\n')
    heading('PA2: Instruction tuning and preference optimization')
    paragraph('Repository: '+(a.repo_url or 'PENDING GITHUB AUTHENTICATION - draft, not submission ready.'))
    paragraph('All comparisons use saved generations and blinded Kaggle model-proxy judgments. '
              'SFT and full-parameter DPO use Qwen/Qwen2.5-0.5B. The separate persona track uses '
              'HuggingFaceTB/SmolLM2-360M-Instruct, compared with its untuned instruction checkpoint.')
    table([
        ['Run / data','Optimizer and schedule','Batch / updates','Parameter strategy'],
        ['SFT: UltraChat first turn, 30,000 train / 1,000 val', 'AdamW; LR 2e-5; cosine; 3% warmup; WD 0.1',
         '2 x accum 16 = 32; 1 epoch; 938 updates', 'Full FP32 weights; FP16 autocast/scaler'],
        ['HH DPO: Anthropic/hh-rlhf, 15,000 / 1,000', 'AdamW; fixed LR 1e-6; no warmup; WD 0.1',
         '1 x accum 32 = 32; 1 epoch; 469 attempts / 466 updates', 'Full FP32 policy; frozen FP16 SFT reference'],
        ['UF DPO: H4/ultrafeedback_binarized, 15,000 / 1,000', 'AdamW; fixed LR 1e-6; no warmup; WD 0.1',
         '1 x accum 32 = 32; 1 epoch; 469 attempts / 466 updates', 'Same SFT best policy/reference initialization'],
        [f'Persona: provided Ramsay preferences, {persona_config["train_count"]} / {persona_config["validation_count"]}',
         'AdamW; LR 5e-5; cosine; 3% warmup; WD 0', '2 x accum 8 = 16; 3 epochs; 39 updates; one T4',
         'LoRA rank 16, alpha 32; q/k/v/o_proj; base frozen'],
    ],[1.65*inch,1.65*inch,1.55*inch,1.6*inch])
    paragraph('Shared settings: seed 42, sequence limit 512, maximum training prompt length 256, '
              'gradient clipping 1.0, DPO beta 0.1. Last partial accumulation windows use their actual size. '
              'SFT averages negative log probability over response tokens, including EOS; prompts and padding '
              'are masked. DPO sums response log probabilities. Prompt template: '
              '### Instruction:\\n{instruction}\\n\\n### Response:\\n. Prompts/responses are tokenized separately.',True)
    paragraph('Both full DPO policy/reference models initialize from the saved SFT best at step 938 '
              '(pa2-sft-training version 2). SFT accumulation averages microbatch token means; validation '
              'is weighted by response-token counts. This differs from a global token mean across an entire '
              'accumulation window when response lengths differ.',True)
    paragraph('SFT extracts the first user/assistant pair. Both full DPO tracks exclude multi-turn pairs, '
              'exclude prompts over 256 tokens and deduplicate prompts across train/validation. '
              'AlpacaEval uses random.Random(42).sample over 300 rows; HH uses the supplied fixed 300 rows. '
              'Persona uses 20 fixed held-out prompts across cooking, coding, general help and safety.',True)
    paragraph('Guide settings were retained for SFT/full DPO. LoRA rank 16 and attention projections provide '
              'moderate style capacity on the small dataset; LR 5e-5 and three epochs form the tested adapter recipe. '
              'Configured LoRA dropout is 0.05, but TRL disable_dropout=True disables dropout during this run. '
              'The persona reference is the base model with adapters disabled; reference scores are precomputed.',True)
    page()
    heading('SFT training behavior and implementation validation')
    figure('sft',6.0,4.2)
    paragraph(f'Validation loss began at {validations["sft"][0]["loss"]:.4f}, rose to '
              f'{validations["sft"][1]["loss"]:.4f} at step 100 and ended at {best["sft"]["loss"]:.4f}. '
              f'The best checkpoint is step {best["sft"]["step"]}. The final decrease is modest; '
              'training loss alone does not establish assistant quality. No SFT updates were skipped.')
    paragraph('The original implementation uses PyTorch loops, gradient accumulation, FP16 autocast with FP32 '
              'trainable weights, gradient scaling, clipping, validation and best/final checkpointing. '
              'Neither Hugging Face Trainer nor TRL is used in SFT or either full DPO run. The six-function '
              'submission_adapter.py exposes batching, response scoring and losses. All seven official public '
              'tests and twelve additional masking, independent scalar-reference, gradient and evaluator '
              'tests pass locally (19 total); eleven earlier core tests also passed in the Kaggle runtime.')
    paragraph('A plain greedy three-prompt sanity check loaded the checkpoints correctly but exposed repetition '
              'and factual errors. Main generation therefore uses deterministic repetition_penalty=1.1 and '
              'no_repeat_ngram_size=4 for every compared model. These controls address looping, not factuality. '
              'Decoding uses do_sample=False, max_new_tokens=256, no length penalty adjustment, and default '
              'temperature/top-p values of 1 (inactive under greedy decoding). Persona uses its native chat template.')
    page()
    heading('HH-RLHF DPO training behavior')
    figure('dpo_hh',6.0,4.5)
    paragraph(f'Best validation reward accuracy: {best["hh"]["reward_accuracy"]:.1%} at step '
              f'{best["hh"]["step"]}, loss {best["hh"]["loss"]:.4f}, reward margin '
              f'{best["hh"]["reward_margin"]:.4f}. Selection uses strict maximum validation reward accuracy. '
              'Both best and final checkpoints are saved. Validation preference diagnostics aggregate all 1,000 pairs; '
              'training loss is averaged across each accumulation window. Training reward diagnostics from the last '
              'microbatch are omitted from the figure.')
    paragraph('The policy uses cuda:0 and the frozen reference cuda:1. Three attempted updates were skipped by '
              'AMP (steps 1, 4, 264), so 466 parameter updates succeeded. These were scaler-protected overflows, '
              'and no non-finite-loss termination occurred. Initial reward accuracy was 51.4% with a near-zero margin: '
              'FP32 policy and FP16 reference weights introduce small scoring differences even before optimization. '
              'This is numerical behavior, not preexisting learned preference improvement.')
    paragraph('The positive final reward margin indicates preference learning on held-out HH pairs. '
              'It does not establish that all refusals are appropriate or that helpfulness improved; those claims '
              'require the saved-output safety comparison on the fixed HH evaluation set.')
    page()
    heading('UltraFeedback DPO training behavior')
    figure('dpo_uf',6.0,4.5)
    paragraph(f'UltraFeedback best reward accuracy is {best["uf"]["reward_accuracy"]:.1%} at step '
              f'{best["uf"]["step"]}; loss {best["uf"]["loss"]:.4f}, margin {best["uf"]["reward_margin"]:.4f}. '
              f'Final accuracy is {validations["uf"][-1]["reward_accuracy"]:.1%}; evaluation uses the best '
              'checkpoint rather than the final one. Three updates were skipped (1, 4, 63). Initial reward '
              'accuracy 47.0% also reflects the mixed-precision reference, not learned preference ability.')
    page()
    heading('Persona LoRA DPO training behavior')
    figure('persona',6.0,4.5)
    paragraph(f'Persona validation reward accuracy reached 100% on its 22 held-out pairs at every epoch. '
              f'The selected checkpoint is step {persona_best_step} (the first maximum); its validation loss '
              f'is {persona_best["eval_loss"]:.4f} and margin {persona_best["eval_rewards/margins"]:.4f}. '
              'Later epochs reduced loss but tied accuracy, so they were not selected. Such a small, stylized '
              'validation set can be easy and is weak evidence of broad usefulness or safety.')
    paragraph('The first persona attempt failed at step zero due to DataParallel/checkpointing tensors spanning '
              'cuda:0 and cuda:1. Restricting the TRL subprocess to one visible T4 resolved it. This resource-driven '
              'change makes the effective persona batch 16. Training completed 39 updates; the restored best '
              'adapter was saved and merged into a standalone generation checkpoint.')
    page()
    heading('Blinded model-judged evaluation')
    table_data=[['Track / comparison','N','W / T / L','Unknown / failed','Win% excl. ties / ties-half']]
    for t in tracks:
        e=evaluations[t]
        rate=lambda k: 'n/a' if e[k] is None else f'{100*e[k]:.1f}'
        table_data.append([t+' vs '+('SmolLM2-Instruct' if t=='persona' else 'SFT'), e['sample_count'],
                           f"{e['wins']} / {e['ties']} / {e['losses']}",f"{e['unknown']} / {e['failures']}",
                           rate('win_rate_excluding_ties')+' / '+rate('win_rate_ties_half')])
    table(table_data,[1.6*inch,0.35*inch,1.1*inch,1.2*inch,2.2*inch])
    paragraph('Judge: google/gemini-3.6-flash via Kaggle-supported model proxy. Response order is randomized '
              'with seed 42; model names live only in a separate identity mapping. Judge prompts contain '
              'instruction and Response A/B only. Batches contain five pairs, with at least five seconds between '
              'request attempts, up to three attempts and append-only checkpoints. Resume skips completed IDs. '
              'Unresolved batches are later retried as individual requests using the same rubric; these decisions '
              'are marked recovered_single_pair in the saved records. '
              'Unknown/invalid/unresolved results do not count as wins. Win% excluding ties = W/(W+L); '
              'ties-half = (W+0.5T)/(W+T+L). No unlike evaluation sets are combined.')
    persona_mapping={r['id']:r for r in rows(generation_file('persona','pairs.identities.jsonl'))}
    persona_judged=rows(evaluation_file('persona','judged.jsonl'))
    rating_table=[['Persona rating (1-5)','Tuned mean','Baseline mean','Complete rating pairs']]
    for dimension in ('style','usefulness','safety'):
        candidate_ratings,baseline_ratings=[],[]
        for judgment in persona_judged:
            mapping=persona_mapping[judgment['id']]
            side='a' if mapping['A']==mapping['candidate'] else 'b'
            other='b' if side=='a' else 'a'
            cr,br=judgment.get(dimension+'_'+side),judgment.get(dimension+'_'+other)
            if isinstance(cr,(float,int)) and isinstance(br,(float,int)) and 1<=cr<=5 and 1<=br<=5:
                candidate_ratings.append(cr)
                baseline_ratings.append(br)
        n=len(candidate_ratings)
        rating_table.append([dimension,f'{sum(candidate_ratings)/n:.2f}' if n else 'not returned',
                             f'{sum(baseline_ratings)/n:.2f}' if n else 'not returned',n])
    table(rating_table,[2.7*inch,1.0*inch,1.0*inch,1.75*inch])
    for t in tracks:
        e=evaluations[t]
        delta=e['wins']-e['losses']
        verdict='candidate won more often' if delta>0 else ('baseline won more often' if delta<0 else 'wins and losses were balanced')
        failfile=evaluation_file(t,'judged.failures.jsonl')
        failures=rows(failfile) if failfile.exists() else []
        auth_count=sum('401' in f.get('error','') for f in failures)
        paragraph(f'{t}: {verdict}, by {abs(delta)} pairs. '
                  f'Historical failed request attempts: {len(failures)}, including {auth_count} authentication failures. '
                  f'Final unresolved pairs: {e["failures"]}; individually recovered pairs: {e.get("recovered_single_pairs",0)}.',True)
    paragraph('Historical authentication failures yielded no judgments and do not count as model losses. '
              'The reported scores use valid saved decisions after credential renewal and parsing retries.',True)
    if evaluations['uf']['wins']<evaluations['uf']['losses']:
        paragraph('UltraFeedback preference validation improved, yet SFT won more AlpacaEval comparisons. '
                  'Thus higher reward accuracy did not translate into general helpfulness here. Preference-data '
                  'mismatch and over-optimization are possible explanations; the examples below provide observed '
                  'behaviors, while the experiment cannot isolate the causal mechanism.',True)
    if evaluations['hh']['wins']>evaluations['hh']['losses']:
        paragraph('HH DPO won more often on the safety-oriented set. This is consistent with the match between '
                  'helpful/harmless preference data and the evaluation target, but does not prove data choice '
                  'was the cause; decoding, checkpoint quality and judge preferences also matter.',True)
    # Representative observed wins/regressions, including judge rationale.
    for t in ('hh','uf'):
        mapping={r['id']:r for r in rows(generation_file(t,'pairs.identities.jsonl'))}
        judged=rows(evaluation_file(t,'judged.jsonl'))
        pair_content={r['id']:r for r in rows(generation_file(t,'pairs.jsonl'))}
        for desired in ('win','loss'):
            found=next((r for r in judged if r['winner'] in ('A','B') and
                        (mapping[r['id']][r['winner']]==mapping[r['id']]['candidate'])==(desired=='win')),None)
            if found:
                pair=pair_content[found['id']]
                mapping_row=mapping[found['id']]
                cand_side='response_a' if mapping_row['A']==mapping_row['candidate'] else 'response_b'
                base_side='response_b' if cand_side=='response_a' else 'response_a'
                paragraph(f"{t} {desired}, ID {found['id']}. Prompt excerpt: {pair['instruction'][:75]}... "
                          f"SFT excerpt: {pair[base_side][:75]}... DPO excerpt: {pair[cand_side][:75]}... "
                          f"Judge rationale excerpt: {found['reason'][:100]}...",True)
    page()
    heading('Interpretation, limitations and reproducibility')
    paragraph('Measured preference learning and response quality are distinct outcomes. HH and UltraFeedback '
              'both show higher held-out preference accuracy and positive reward margins; their judge comparisons '
              'above determine whether that translated to better saved responses on each target set. Data mismatch '
              'between training pairs and evaluation prompts, label noise, decoding effects and the limited capacity '
              'of a 0.5B model are plausible explanations for regressions, but this experiment does not isolate '
              'those causes. The three raw sanity responses already showed repetition and factual weaknesses '
              'in the SFT baseline. No multi-seed or beta ablation was run.')
    paragraph('Persona evaluates a different target: style fidelity together with usefulness and safety. Its '
              '20-prompt sample and 22-pair validation set are small. The judge separately records style/usefulness/'
              'safety ratings for A/B where returned; those ratings should be inspected alongside examples rather '
              'than treating style preference as evidence of general capability. Model-judge bias, position bias '
              'despite randomization, single-seed variance, partial truncation at 512 tokens and repeated judge '
              'attempts limit the conclusions. Exact train/validation prompts are disjoint within each track. '
              'The custom persona evaluation has no '
              'exact prompt matches in the supplied training file; related cooking topics can still overlap.')
    if data_audit:
        paragraph(f'A separate saved CPU Kaggle audit verified all six actual subsets: counts, SHA-256 hashes, '
                  f'prompt uniqueness, train/validation separation, the template and <=256 prompt tokens. '
                  'Two UF training prompts exactly match the fixed AlpacaEval set; that score is therefore '
                  'not wholly unseen. HH validation has 14 AlpacaEval matches and HH training one normalized '
                  'match, outside its HH target evaluation. No SFT/evaluation or full-DPO/HH-target matches '
                  'were found. Detailed counts are in results/audit/data_audit.json.',True)
    if evaluations['persona']['ties']>=evaluations['persona']['sample_count']/2:
        paragraph(f"Persona had {evaluations['persona']['ties']} ties out of {evaluations['persona']['sample_count']} prompts. "
                  'The separate rating table reports whether style, usefulness or safety changed; in this run '
                  'there is little visible style steering despite easy preference validation. The first accuracy '
                  'maximum selected only 13 updates, while validation loss continued improving later. Accuracy '
                  'saturation on 22 stylized pairs may therefore have made checkpoint selection insensitive to '
                  'further style learning. This is a plausible interpretation, not an ablation result.',True)
    paragraph('Training/generation ran on private Kaggle notebooks using two T4s (one visible T4 for persona). '
              'The primary 620 judgments ran locally through the supported Kaggle proxy. '
              + ('A saved private utility benchmark then resumed and verified all 620, and separately judged '
                 'the first five saved pairs per track with Gemini 3.6 (15 fresh audit pairs). These audit decisions '
                 'do not replace primary scores. Utility: https://www.kaggle.com/benchmarks/tasks/umarmazhar2025/pa2-saved-pairwise-evaluation/2. '
                 'Kaggle also automatically validated task creation with Gemini 3.8; its outputs remain separate. '
                 if utility else 'The saved Kaggle utility verification remains pending. ')
              + 'The initial plain SFT sanity was inspected before choosing decoding controls; controlled '
              'DPO/persona sanity cells originally proceeded automatically. Their manual review occurred '
              'during the final audit; revised notebooks require review before full generation. This timing '
              'deviation cannot be retroactively corrected. No credentials are included.',True)
    paragraph('Reproduction: scripts/prepare_data.py; scripts/train.py; scripts/train_persona.py; '
              'scripts/generate.py; scripts/build_pairs.py; scripts/judge_pairwise_kaggle.py. '
              'Run python -m pytest -q tests. Original run configs/metrics are under outputs/monitor/<track>/artifacts; '
              'saved generations and blinded pairs under outputs/generated/<track>/pa2/outputs/eval/<track>; '
              'judgments/failure logs under outputs/evaluation/generate-<track>. Notebook versions preserve '
              'best/final weights and exact subsets. Source dataset fingerprints and hashes are in their '
              'outputs/data/manifest_<track>.json files. Submission packaging copies this evidence into results/.',True)
    heading('References')
    paragraph('Saved checkpoint notebooks: https://www.kaggle.com/code/umarmazhar2025/pa2-sft-training; '
              'https://www.kaggle.com/code/umarmazhar2025/pa2-dpo-hh; '
              'https://www.kaggle.com/code/umarmazhar2025/pa2-dpo-uf; '
              'https://www.kaggle.com/code/umarmazhar2025/pa2-persona-lora (version 2).',True)
    for reference in [
        'Rafailov et al. (2023), Direct Preference Optimization. https://arxiv.org/abs/2305.18290',
        'Qwen2.5-0.5B model card: https://huggingface.co/Qwen/Qwen2.5-0.5B',
        'SmolLM2-360M-Instruct: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct',
        'Data: HuggingFaceH4/ultrachat_200k; HuggingFaceH4/ultrafeedback_binarized; Anthropic/hh-rlhf; tatsu-lab/alpaca_eval (Hugging Face dataset cards).',
        'TRL v0.24 DPOTrainer: https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer; PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora',
        'Kaggle Benchmarks model proxy: https://github.com/Kaggle/kaggle-benchmarks; starter: https://github.com/ahmad903501/small-model-alignment-starter',
    ]:
        paragraph(reference,True)
    page()
    heading('Prompt-type breakdown and observed failure cases')
    category_file=audit_root/'category_results.json'
    if not category_file.exists():
        raise RuntimeError('Run scripts.audit_evidence before generating the final report')
    categories=json.loads(category_file.read_text())
    table([['HH category (N=30 each)','DPO W / T / L','Win% excluding ties']]+[
        [category.replace('_',' '), f"{s['wins']} / {s['ties']} / {s['losses']}",
         f"{s['win_rate_excluding_ties']:.1%}"] for category,s in categories['hh'].items()
    ],[3.6*inch,1.4*inch,1.45*inch])
    paragraph('HH benefits are uneven: cyber abuse and legal/financial prompts favor DPO, while self-harm '
              'support favors SFT (12 versus 18 decisive wins). Safe transformation is tied. Relative '
              'preference wins do not establish absolute safety: manually reviewed harmful-request IDs '
              'hh-eval-42-0000 and 0001 still receive unsafe assistance from HH DPO.',True)
    table([['AlpacaEval source subset','N','UF DPO W / T / L']]+[
        [category,s['sample_count'],f"{s['wins']} / {s['ties']} / {s['losses']}"]
        for category,s in categories['uf'].items()
    ],[3.2*inch,0.8*inch,2.45*inch])
    paragraph('UF loses more than it wins in every source subset; the helpful_base gap is largest '
              '(9 wins, 32 losses). Visible factuality failures include ID 654 inventing 100+ dialogue '
              'domains where the supplied abstract states 59, and ID 25 listing living species and unrelated '
              'animals as extinct bears. These defects support the observed lack of helpfulness gain.',True)
    sensitivity_file=audit_root/'evaluation_overlap_sensitivity.json'
    if sensitivity_file.is_file():
        sensitivity=json.loads(sensitivity_file.read_text())
        e=sensitivity['excluding_training_matches']
        paragraph(f'Overlap sensitivity (not a replacement for the prescribed set): excluding the two '
                  f'UF training-prompt matches leaves N={e["sample_count"]}, W/T/L '
                  f'{e["wins"]}/{e["ties"]}/{e["losses"]}, decisive win rate '
                  f'{e["win_rate_excluding_ties"]:.1%}. The conclusion remains that SFT wins more often.',True)
    paragraph('Persona W/T/L by category (N=5 each): cooking 2/3/0, coding 0/4/1, general 1/3/1, safety '
              '0/4/1. Both models average style 1/5. The tuned scrambled-egg response begins with an '
              'unnecessary capability disclaimer and truncates before finishing its advice; the dough '
              'response is generic rather than the requested blunt coach. Cooking preferences alone '
              'therefore do not demonstrate successful broad persona alignment.',True)
    name='PA2_report.pdf' if a.repo_url else 'PA2_report_DRAFT.pdf'
    doc=SimpleDocTemplate(str(output/name),pagesize=(8.5*inch,11*inch),
                         leftMargin=0.75*inch,rightMargin=0.7*inch,topMargin=0.55*inch,bottomMargin=0.5*inch)
    def footer(canvas, document):
        canvas.setFont('Helvetica',8)
        canvas.drawRightString(7.8*inch,0.25*inch,str(document.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(output/name).pages)
    if count>8:
        raise RuntimeError(f'Report has {count} pages; shorten before submission')
    (output/'PA2_report.md').write_text('\n'.join(markdown),encoding='utf-8')
    (output/'report_status.json').write_text(json.dumps({'pages':count,'repository_url':a.repo_url,
                                                       'submission_ready':bool(a.repo_url and utility and data_audit),
                                                       'native_utility_verified':bool(utility),
                                                       'saved_data_audited':bool(data_audit)},indent=2))
    print(f'Wrote {name}, {count} pages')


if __name__=='__main__':
    main()
