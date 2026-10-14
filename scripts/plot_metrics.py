"""Export readable unsmoothed training and validation curves."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--output', default='outputs/plots')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for run in a.runs:
        run = Path(run)
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
        metrics = ['loss'] + (['reward_accuracy','reward_margin'] if any('reward_margin' in r for r in rows) else [])
        validation=[r for r in rows if r['phase']=='validation']
        selected=(max(validation,key=lambda r:r['reward_accuracy']) if len(metrics)>1
                  else min(validation,key=lambda r:r['loss'])) if validation else None
        columns = 1 if len(metrics)==1 else 2
        fig, axes = plt.subplots(math.ceil(len(metrics)/columns), columns,
                                 figsize=(5,3.5) if len(metrics)==1 else (8,6), squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            # Training preference diagnostics currently describe the last
            # microbatch; validation diagnostics aggregate the entire split.
            for phase in (('train','validation') if metric=='loss' else ('validation',)):
                points = [r for r in rows if r['phase']==phase and metric in r]
                if points:
                    ax.plot([r['step'] for r in points],[r[metric] for r in points],label=phase)
            ax.set(xlabel='Optimizer step',ylabel=metric.replace('_',' '),title=run.name)
            if selected is not None:
                ax.axvline(selected['step'],color='black',linestyle='--',alpha=.5,
                           label=f"best step {selected['step']}")
            ax.legend()
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(metrics):]:
            ax.set_visible(False)
        fig.suptitle('Unsmoothed metrics')
        fig.tight_layout()
        fig.savefig(out/f'{run.name}.png', dpi=180)
        fig.savefig(out/f'{run.name}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()
