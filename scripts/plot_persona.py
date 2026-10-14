"""Plot actual TRL log history for the persona run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source=Path('outputs/monitor/persona_v2/artifacts/pa2/outputs/persona/metrics.json')
    data=json.loads(source.read_text())
    completed=json.loads((source.parent/'completed.json').read_text())
    selected=int(completed['best_checkpoint'].rsplit('-',1)[1])
    fig,axes=plt.subplots(2,2,figsize=(8,6))
    for row in (('loss','eval_loss','Loss'),('rewards/accuracies','eval_rewards/accuracies','Reward accuracy'),
                ('rewards/margins','eval_rewards/margins','Reward margin')):
        ax=axes.flat[list(('Loss','Reward accuracy','Reward margin')).index(row[2])]
        for key,label in zip(row[:2],('train','validation')):
            points=[r for r in data if key in r]
            if points:
                ax.plot([r['step'] for r in points],[r[key] for r in points],label=label)
        ax.set(xlabel='Optimizer step',ylabel=row[2],title='Persona LoRA DPO')
        ax.axvline(selected,color='black',linestyle='--',alpha=.5,label=f'best step {selected}')
        ax.grid(alpha=.2)
        ax.legend()
    axes.flat[3].set_visible(False)
    fig.suptitle(f'Unsmoothed TRL log history; selected checkpoint: step {selected}')
    fig.tight_layout()
    target=Path('outputs/plots')
    target.mkdir(parents=True,exist_ok=True)
    fig.savefig(target/'persona.png',dpi=180)
    fig.savefig(target/'persona.pdf')
