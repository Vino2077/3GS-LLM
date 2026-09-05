"""Reproducible mixture. Repeat draws are explicitly distinguished from unique pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from prepare_sft import records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('authored', type=Path)
    ap.add_argument('conversation', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--reviewed-teacher', type=Path)
    ap.add_argument('--general-repeats', type=int, default=5)
    ap.add_argument('--teacher-repeats', type=int, default=1)
    ap.add_argument('--general-only', action='store_true')
    a = ap.parse_args()
    base = list(records(a.authored))
    general = [r for r in base if r['source'] != 'authored-game-alignment'] + list(records(a.conversation))
    games = [r for r in base if r['source'] == 'authored-game-alignment']
    teacher = list(records(a.reviewed_teacher)) if a.reviewed_teacher else []
    if any(r.get('review_status') != 'accepted' for r in teacher):
        raise ValueError('Only reviewed teacher pairs may enter the mixture')
    rows = ([] if a.general_only else games) + general * a.general_repeats + teacher * a.teacher_repeats
    random.Random(20260905).shuffle(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    report = dict(unique_pairs=len({(r['prompt'],r['response']) for r in rows}),
                  draws=len(rows), draws_by_source=dict(Counter(r['source'] for r in rows)),
                  sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(), seed=20260905,
                  warning='Template expansion and repeated sampling are not independent teacher examples.')
    a.output.with_suffix('.stats.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
