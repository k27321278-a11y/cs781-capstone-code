"""One-shot official-dev evaluation of calibration-selected sparse candidate."""
import json
import pickle
import time

import numpy as np

from run_sparse import DATA, OUT, read_rows, score


def main():
    t0=time.time()
    with (OUT / "logistic_model.pkl").open("rb") as f: a=pickle.load(f)
    dev=read_rows("dev")
    y=np.zeros((len(dev),100),dtype=np.uint8)
    for k,r in enumerate(dev): y[k,r["labels"]]=1
    fresh=a["model"].predict_proba(a["vectorizer"].transform(r["text"] for r in dev)).astype(np.float32)
    if a["candidate"] == "blend_75":
        prior=a["baseline"].predict_proba(a["baseline_vectorizer"].transform(r["text"] for r in dev)).astype(np.float32)
        probs=.75*fresh+.25*prior
    elif a["candidate"] == "blend_50":
        prior=a["baseline"].predict_proba(a["baseline_vectorizer"].transform(r["text"] for r in dev)).astype(np.float32)
        probs=.5*fresh+.5*prior
    else: probs=fresh
    pred=(2*probs-1)>=a["thresholds"]
    report={"selection":a["candidate"],"selection_basis":"official-train calibration only",
            "official_dev_examples":len(dev),"labels":100,"metrics":score(y,pred),
            "positive_predictions":int(pred.sum()),"elapsed_seconds":round(time.time()-t0,1)}
    (OUT / "dev_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
