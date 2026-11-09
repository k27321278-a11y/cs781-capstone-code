import json, statistics
from pathlib import Path
root=Path('experiments/legalspan_coverage')
r=json.loads((root/'outputs/coverage_pilot/report.json').read_text(encoding='utf-8'))
a={x['id']:x for x in r['runs'] if x['strength']==0.}
b={x['id']:x for x in r['runs'] if x['strength']==.5}
assert len(a)==len(b)==8 and set(a)==set(b)
summary={'n':8,'generated_token_cap':384,'same_text_count':sum(a[i]['summary']==b[i]['summary'] for i in a),'metrics':{}}
for metric in ['word_count','unigram_overlap_f1_proxy','repeated_trigram_fraction']:
 av=statistics.mean(x['metrics'][metric] for x in a.values());bv=statistics.mean(x['metrics'][metric] for x in b.values())
 summary['metrics'][metric]={'baseline':av,'coverage':bv,'difference':bv-av}
(root/'pilot_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
for i in ['837','3472']:
 print('DOCUMENT',i)
 print('BASELINE',a[i]['summary'])
 print('COVERAGE',b[i]['summary'])
 print('REFERENCE',a[i]['reference'][:3000])

