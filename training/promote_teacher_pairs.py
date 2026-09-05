"""Deterministic final gate for locally generated Stage-3 pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from prepare_sft import records
from stage3_quality import response_rejection

BAD = re.compile(r'\b(уведомим|килограмм|кило|точно будет|гарантир|как ии|языковая модель)\b', re.I)
MANUAL_REJECT = {
    # Fixed review of the 400-row Qwen3.5-27B run: unsupported facts,
    # misunderstood numbers, invented personal experience, or topic leakage.
    'dtf-a75b6b7403bc0b114d0e15bf', 'dtf-1617f9bba5393b905c01f2ce',
    'dtf-1453220daabe9e44bef0f880', 'dtf-fd2493a87e2998bd250914c9',
    'dtf-f4101b561f9f6629285f94cd', 'dtf-3b63f7a0996fa489ceb4cc81',
    'dtf-354d660dd44870c60b558d7f', 'dtf-14e75971457c42fbe4bf768d',
    'dtf-9472423546ce4985496f9149',
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',type=Path)
    ap.add_argument('output',type=Path)
    a=ap.parse_args()
    out=[]; stats=Counter()
    for row in records(a.input):
        stats['input']+=1
        score=row.get('judge',{})
        answer=str(row.get('response','')).strip()
        if row.get('status')!='accepted':
            stats['not_accepted']+=1; continue
        if any(score.get(k)!=5 for k in ('r','d','f','n')) or score.get('s',0)<3:
            stats['not_all_core_fives']+=1; continue
        if row['id'] in MANUAL_REJECT or BAD.search(answer) or response_rejection(row['prompt'],answer):
            stats['rule_rejected']+=1; continue
        promoted={k:v for k,v in row.items() if k not in ('candidates','generation_usage','judge_usage')}
        promoted['review_status']='accepted'
        promoted['source']='local-qwen35-27b-distillation'
        out.append(promoted)
    stats['promoted']=len(out)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),encoding='utf-8')
    a.output.with_suffix('.stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
