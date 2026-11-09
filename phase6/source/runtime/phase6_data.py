"""Public-input handling and deterministic document-only passage coverage."""
import hashlib
import re


def normalize_rows(data, training, drop_empty_summaries=False):
    rows=[]
    items=data.items() if isinstance(data,dict) else ((None,r) for r in data)
    for key,value in items:
        if isinstance(value,str): value={'document':value}
        if not isinstance(value,dict): raise ValueError('Expected document objects')
        row=dict(value)
        row['id']=str(row.get('id',key))
        if key is None and 'id' not in value: raise ValueError('Missing official ID')
        if not isinstance(row.get('document'),str) or not row['document'].strip():
            raise ValueError('Missing document for '+row['id'])
        if training and (not isinstance(row.get('summary'),str) or not row['summary'].strip()):
            if drop_empty_summaries: continue
            raise ValueError('Missing training summary')
        if not training and 'summary' in row: raise ValueError('Unexpected test summary')
        rows.append(row)
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate input IDs')
    if not rows: raise ValueError('Empty official input')
    return rows


def document_hash(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text).strip().casefold().encode()).hexdigest()


def split_rows(rows):
    splits={'fit':[],'validation':[],'confirmation':[]}
    for row in rows:
        bucket=int(document_hash(row['document'])[:12],16)%100
        split='fit' if bucket<90 else 'validation' if bucket<98 else 'confirmation'
        splits[split].append(row)
    return splits


def select_passages(tokens,count=8,width=384):
    if not tokens: raise ValueError('Document tokenization is empty')
    if len(tokens)<=width: return [tokens]
    if len(tokens)<=count*width:
        return [tokens[i:i+width] for i in range(0,len(tokens),width)]
    if count<2: return [tokens[:width]]
    starts=[round(i*(len(tokens)-width)/(count-1)) for i in range(count)]
    return [tokens[start:start+width] for start in starts]


def validate_predictions(predictions,expected_ids):
    ids=[r['id'] for r in predictions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):
        raise ValueError('Missing, extra or duplicate prediction IDs')
    if any(not isinstance(r.get('summary'),str) or not r['summary'].strip() for r in predictions):
        raise ValueError('Empty or nonstring summary')
    return True
