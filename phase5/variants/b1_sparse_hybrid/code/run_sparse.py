"""Official-data-only sparse legal classifier experiment."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "phase5_data"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)


def read_rows(name):
    with (DATA / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)


def score(y, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0))}


def tune_thresholds(y, z):
    grid = np.arange(-2., 2.001, .05)
    tp = ((z[:, :, None] >= grid) & y[:, :, None]).sum(axis=0)
    pp = (z[:, :, None] >= grid).sum(axis=0)
    support = y.sum(axis=0)
    f1 = np.divide(2 * tp, pp + support[:, None], out=np.zeros_like(tp, dtype=float), where=(pp+support[:, None]) > 0)
    global_score = f1.mean(axis=0)
    global_t = float(grid[global_score.argmax()])
    local_t = grid[f1.argmax(axis=1)]
    wt = support / (support + 40)
    thresholds = wt * local_t + (1-wt) * global_t
    return thresholds.astype(np.float32), global_t


def main():
    t0=time.time()
    manifest=json.load((DATA / "split_manifest.json").open(encoding="utf-8"))
    train=read_rows("train")
    ids={r["id"]:r for r in train}
    fit=[ids[i] for i in manifest["fit_ids"]]
    cal=[ids[i] for i in manifest["calibration_ids"]]
    yfit=np.zeros((len(fit),100),dtype=np.uint8)
    ycal=np.zeros((len(cal),100),dtype=np.uint8)
    for k,r in enumerate(fit): yfit[k,r["labels"]]=1
    for k,r in enumerate(cal): ycal[k,r["labels"]]=1
    with (ROOT / "work" / "phase5_sgd_fulltext.pkl").open("rb") as f:
        base_vect, base, base_t=pickle.load(f)
    print("data/baseline loaded",round(time.time()-t0,1),flush=True)
    bcal=base.predict_proba(base_vect.transform(r["text"] for r in cal)).astype(np.float32)
    vect=TfidfVectorizer(dtype=np.float32,max_features=160000,min_df=3,max_df=.98,ngram_range=(1,2),sublinear_tf=True)
    Xfit=vect.fit_transform(r["text"] for r in fit)
    Xcal=vect.transform(r["text"] for r in cal)
    print("features",Xfit.shape,Xfit.nnz,round(time.time()-t0,1),flush=True)
    model=OneVsRestClassifier(SGDClassifier(loss="modified_huber",alpha=3e-5,class_weight="balanced",max_iter=8,tol=1e-3,random_state=781),n_jobs=2)
    model.fit(Xfit,yfit)
    raw=model.decision_function(Xcal).astype(np.float32)
    print("new model fit",round(time.time()-t0,1),flush=True)
    # Modified Huber has a common [-1, 1] probability-like score range.
    # Compare direct model and a lightweight average with the earlier logistic model.
    candidates={"modified_huber":raw,"hybrid_50":0.5*raw+0.5*(2*bcal-1),"hybrid_25":0.25*raw+0.75*(2*bcal-1)}
    reports={}
    for key,z in candidates.items():
        thresh, global_t=tune_thresholds(ycal,z)
        pred=z>=thresh
        reports[key]={**score(ycal,pred),"global_threshold":global_t,"thresholds":thresh.tolist(),"positive_predictions":int(pred.sum())}
        print(key,reports[key]["macro_f1"],reports[key]["micro_f1"],flush=True)
    best=max(reports,key=lambda key:reports[key]["macro_f1"])
    artifact={"vectorizer":vect,"model":model,"baseline_vectorizer":base_vect if best.startswith("hybrid") else None,"baseline":base if best.startswith("hybrid") else None,
              "candidate":best,"thresholds":np.array(reports[best]["thresholds"],dtype=np.float32)}
    with (OUT / "selected_model.pkl").open("wb") as f: pickle.dump(artifact,f,pickle.HIGHEST_PROTOCOL)
    report={"method":"official train fit, train calibration; fresh fit-only full-document word TF-IDF",
            "counts":{"fit":len(fit),"calibration":len(cal),"labels":100},"candidate_reports":reports,
            "selection":best,"elapsed_seconds":round(time.time()-t0,1)}
    (OUT / "calibration_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("selected",best,"elapsed",round(time.time()-t0,1),flush=True)


if __name__=="__main__": main()
