"""Materialize a fixed stratified audit; never invent automatic semantic scores."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--preset',default='A')
    ap.add_argument('--models',nargs='+',default=['old','aligned-v2','A','B','C','D','E','F'])
    ap.add_argument('--per-category',type=int,default=2)
    a=ap.parse_args()
    by_id={}
    hashes={}
    for name in a.models:
        path=a.root/f'eval-{name}'/'responses.jsonl'
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['preset']==a.preset and r['seed']==42 and int(r['id'].rsplit('-',1)[1])<a.per_category:
                item=by_id.setdefault(r['id'],dict(id=r['id'],prompt=r['prompt'],category=r['category'],answers={}))
                item['answers'][name]=r['response']
    out=a.root/f'manual-audit-{a.preset}.json'
    out.write_text(json.dumps(dict(hashes=hashes,selection='first fixed IDs per category, seed42',
        preset=a.preset,items=list(by_id.values())),ensure_ascii=False,indent=2),encoding='utf-8')
    for item in by_id.values():
        print(item['id']+' | '+item['prompt'])
        for name,answer in item['answers'].items():
            print(name+': '+answer)
        print()

if __name__=='__main__':
    main()
