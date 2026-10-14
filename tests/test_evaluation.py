import json
import pytest
from alignment.evaluation import judge_prompt, parse_judgments, summarize


PAIR = {'id':'1','instruction':'Do a thing','response_a':'good','response_b':'bad',
        'generator_a':'secret_model_name'}


def test_judge_prompt_excludes_hidden_names():
    assert 'secret_model_name' not in judge_prompt([PAIR])


def test_invalid_or_duplicate_decisions_rejected():
    with pytest.raises(ValueError):
        parse_judgments(json.dumps([{'id':'1','winner':'C','reason':'bad'}]),[PAIR])
    with pytest.raises(ValueError):
        parse_judgments(json.dumps([{'id':'1','winner':'A','reason':'x'}]*2),[PAIR])


def test_mapping_and_failure_denominators():
    pairs = [{**PAIR,'id':str(i)} for i in range(5)]
    mapping = [{'id':str(i),'A':'candidate','B':'baseline','candidate':'candidate'} for i in range(5)]
    judgments = [{'id':str(i),'winner':w} for i,w in enumerate(['A','B','tie','unknown'])]
    s = summarize(pairs,judgments,mapping)
    assert (s['wins'],s['losses'],s['ties'],s['unknown'],s['failures']) == (1,1,1,1,1)
    assert s['win_rate_excluding_ties']==0.5
    assert s['win_rate_ties_half']==0.5
    assert s['rates_over_all_pairs']['failures']==0.2


def test_persona_requires_complete_integer_ratings():
    row = {'id': '1', 'winner': 'tie', 'reason': 'Similar outputs'}
    with pytest.raises(ValueError):
        parse_judgments(json.dumps([row]), [PAIR], 'persona')
    row.update({dimension+'_'+side: 3 for dimension in ('style','usefulness','safety') for side in ('a','b')})
    assert parse_judgments(json.dumps([row]), [PAIR], 'persona') == [row]
    row['style_a'] = True
    with pytest.raises(ValueError):
        parse_judgments(json.dumps([row]), [PAIR], 'persona')
