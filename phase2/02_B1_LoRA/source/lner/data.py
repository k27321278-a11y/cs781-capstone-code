import bisect
import hashlib
import json
import math
import os
import random
import zipfile
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def byte_offsets_to_chars(text, offsets):
    boundaries = [0]
    for c in text:
        boundaries.append(boundaries[-1]+len(c.encode('utf-8')))
    result = []
    for a,b in offsets:
        if not 0 <= a < b <= boundaries[-1]:
            raise ValueError('Invalid byte offset')
        result.append((bisect.bisect_right(boundaries,a)-1,
                       bisect.bisect_left(boundaries,b)))
    return result


def bio_targets(text, spans, labels):
    output = [0]*len(text)
    for a,b,label,*rest in sorted(spans):
        if not 0 <= a < b <= len(text) or label not in labels:
            raise ValueError('Invalid gold character span')
        if rest and text[a:b] != rest[0]:
            raise ValueError('Gold entity text does not match original characters')
        if any(output[a:b]):
            raise ValueError('Overlapping gold entities need a multilabel head')
        idx = 1+2*labels.index(label)
        output[a] = idx
        output[a+1:b] = [idx+1]*(b-a-1)
    return output


def decode_bio(tags, labels):
    result, start, kind = [], None, None
    for pos,tag in enumerate(list(tags)+[0]):
        label = None if tag == 0 else labels[(int(tag)-1)//2]
        begins = tag > 0 and tag % 2 == 1
        if start is not None and (label != kind or begins):
            result.append([start,pos,kind])
            start,kind = None,None
        if tag and start is None:
            start,kind = pos,label
    return result


def strict_score(truth, predictions, labels):
    if set(truth) != set(predictions):
        raise ValueError('Scoring ID mismatch')
    totals = {label:[0,0,0] for label in labels}
    for key in truth:
        gold = {tuple(s[:3]) for s in truth[key]}
        pred = {tuple(s[:3]) for s in predictions[key]}
        for label in labels:
            g = {s for s in gold if s[2] == label}
            p = {s for s in pred if s[2] == label}
            for k,n in enumerate((len(g&p),len(p-g),len(g-p))):
                totals[label][k] += n
    per_label = {}
    for label,(tp,fp,fn) in totals.items():
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        per_label[label] = {'precision':precision,'recall':recall,'f1':f1,
                            'tp':tp,'fp':fp,'fn':fn}
    return {'macro_f1':sum(x['f1'] for x in per_label.values())/len(labels),
            'macro_precision':sum(x['precision'] for x in per_label.values())/len(labels),
            'macro_recall':sum(x['recall'] for x in per_label.values())/len(labels),
            'per_label':per_label}


def validate_predictions(predictions, texts, labels):
    if not isinstance(predictions,dict) or set(predictions) != set(texts):
        raise ValueError('Prediction IDs do not match official document IDs')
    for key,spans in predictions.items():
        if not isinstance(spans,list):
            raise ValueError('Every document must map to a list')
        seen = set()
        for span in spans:
            if not isinstance(span,list) or len(span) != 3:
                raise ValueError('Expected [start,end,label]')
            a,b,label = span
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(texts[key]):
                raise ValueError('Invalid character boundary')
            if label not in labels or tuple(span) in seen:
                raise ValueError('Unknown label or duplicate span')
            seen.add(tuple(span))


def export_zip(path, predictions, texts, labels):
    validate_predictions(predictions,texts,labels)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('prediction.json',json.dumps(predictions,ensure_ascii=False).encode('utf-8'))
    with zipfile.ZipFile(tmp) as archive:
        assert archive.namelist() == ['prediction.json'] and archive.testzip() is None
        validate_predictions(json.loads(archive.read('prediction.json')),texts,labels)
    os.replace(tmp,path)
    return sha256(path)


def load_dataset(root, seed=781, validation_docs=12):
    root = Path(root)
    labels = (root/'Train/labels.txt').read_text().split()
    if len(labels) != 12 or len(set(labels)) != 12:
        raise ValueError('Expected 12 official labels')
    texts,gold,heldout = {},{},[]
    for fold in (1,2):
        x = json.loads((root/f'Train/fold{fold}-text.json').read_text())
        y = json.loads((root/f'Train/fold{fold}-gold.json').read_text())
        if set(x) != set(y) or set(x)&set(texts):
            raise ValueError('Fold IDs mismatch or duplicate documents')
        keys = sorted(x)
        random.Random(seed+fold).shuffle(keys)
        heldout.extend(keys[:validation_docs//2])
        texts.update(x); gold.update(y)
    test = json.loads((root/'Test/fold3-text.json').read_text())
    if set(texts)&set(test):
        raise ValueError('Official test ID overlaps training')
    targets = {key:bio_targets(text,gold[key],labels) for key,text in texts.items()}
    train_ids = sorted(set(texts)-set(heldout))
    vocab = {c:i+2 for i,c in enumerate(sorted(set(''.join(texts[k] for k in train_ids))))}
    return {'texts':texts,'gold':gold,'test':test,'labels':labels,'targets':targets,
            'train_ids':train_ids,'validation_ids':sorted(heldout),'char_vocab':vocab}


class OffsetTokenizer:
    def __init__(self, path):
        record = json.loads(Path(path).read_text())
        self.own = record.get('format') == 'cs781-own-byte-bpe-v1'
        if self.own:
            from legalspan.bpe import OwnBPE
            self.tokenizer = OwnBPE.load(path)
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self,text):
        enc = self.tokenizer.encode(text)
        offsets = byte_offsets_to_chars(text,enc.offsets) if self.own else enc.offsets
        ids = enc.ids
        if not ids or any(not 0 <= a < b <= len(text) for a,b in offsets):
            raise ValueError('Tokenizer offsets do not preserve source characters')
        mapping = [-1]*len(text)
        roles = []
        for i,(a,b) in enumerate(offsets):
            fragment = text[a:b]
            role = 1 if fragment.strip().isdigit() else 3 if fragment[:1].isupper() else 4 if any(not c.isalnum() and not c.isspace() for c in fragment) else 0
            roles.append(role)
            mapping[a:b] = [i]*(b-a)
        # ByteLevel offsets can omit an isolated space; retain its character position.
        previous = 0
        for i,v in enumerate(mapping):
            if v < 0: mapping[i] = previous
            else: previous = v
        if len(mapping) != len(text) or max(mapping) >= len(ids):
            raise ValueError('Character/token mapping failed')
        if self.own:
            from legalspan.data import role_ids
            roles = role_ids(text,enc)
        return {'ids':ids,'mapping':mapping,'roles':roles}


def prepare_windows(dataset, tokenizers, max_chars=1600, overlap=240, max_tokens=768):
    result = {'train':[], 'validation':[], 'test':[]}
    for group,keys in [('train',dataset['train_ids']),('validation',dataset['validation_ids']),('test',sorted(dataset['test']))]:
        for key in keys:
            text = dataset['test'][key] if group == 'test' else dataset['texts'][key]
            start = 0
            while start < len(text):
                end = min(len(text), start+max_chars)
                while True:
                    fragment = text[start:end]
                    enc = {name:tok.encode(fragment) for name,tok in tokenizers.items()}
                    longest = max(len(v['ids']) for v in enc.values())
                    if longest <= max_tokens: break
                    end = start+max(1,int((end-start)*max_tokens/longest*.95))
                if end <= start: raise ValueError('Window preparation cannot progress')
                row = {'id':key,'start':start,'end':end,'encodings':enc,
                       'chars':[dataset['char_vocab'].get(c,1) for c in fragment]}
                if group != 'test': row['targets'] = dataset['targets'][key][start:end]
                result[group].append(row)
                if end == len(text): break
                start = end-min(overlap,(end-start)//3)
    return result


def constrained_decode(logits, labels, outside_bias=0.):
    import numpy as np
    logits = np.array(logits,dtype=np.float32,copy=True)
    logits[:,0] += outside_bias
    length,classes = logits.shape
    if length==0: return []
    transition = np.zeros((classes,classes),dtype=np.float32)
    for tag in range(2,classes,2):
        transition[:,tag] = -1e4
        transition[tag-1,tag] = transition[tag,tag] = 0.
    score = logits[0].copy(); score[2::2] = -1e4
    back = np.zeros((length,classes),dtype=np.uint8)
    for pos in range(1,length):
        candidates = score[:,None]+transition
        back[pos] = candidates.argmax(0)
        score = candidates.max(0)+logits[pos]
    state = int(score.argmax()); tags = [state]
    for pos in range(length-1,0,-1):
        state = int(back[pos,state]); tags.append(state)
    return decode_bio(tags[::-1],labels)
