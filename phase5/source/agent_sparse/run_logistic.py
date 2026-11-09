"""Second calibration-only experiment with fresh full-document vocabulary."""
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from run_sparse import DATA, OUT, ROOT, read_rows, score, tune_thresholds


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
    with (OUT / "selected_model.pkl").open("rb") as f: old=pickle.load(f)
    vect=old["vectorizer"]
    Xfit=vect.transform(r["text"] for r in fit)
    Xcal=vect.transform(r["text"] for r in cal)
    print("features",Xfit.shape,Xfit.nnz,round(time.time()-t0,1),flush=True)
    model=OneVsRestClassifier(SGDClassifier(loss="log_loss",alpha=1e-5,class_weight="balanced",max_iter=12,tol=1e-3,random_state=781),n_jobs=2)
    model.fit(Xfit,yfit)
    probs=model.predict_proba(Xcal).astype(np.float32)
    print("fitted",round(time.time()-t0,1),flush=True)
    bprobs=old["baseline"].predict_proba(old["baseline_vectorizer"].transform(r["text"] for r in cal)).astype(np.float32)
    candidates={"fresh_logistic":probs,"blend_50":.5*probs+.5*bprobs,"blend_75":.75*probs+.25*bprobs}
    reports={}
    for name,p in candidates.items():
        thresholds,global_t=tune_thresholds(ycal,2*p-1)
        pred=(2*p-1)>=thresholds
        reports[name]={**score(ycal,pred),"global_threshold":global_t,"thresholds":thresholds.tolist(),"positive_predictions":int(pred.sum())}
        print(name,reports[name]["macro_f1"],reports[name]["micro_f1"],flush=True)
    best=max(reports,key=lambda k:reports[k]["macro_f1"])
    artifact={"vectorizer":vect,"model":model,"baseline_vectorizer":old["baseline_vectorizer"] if best.startswith("blend") else None,
              "baseline":old["baseline"] if best.startswith("blend") else None,
              "candidate":best,"thresholds":np.array(reports[best]["thresholds"],dtype=np.float32)}
    with (OUT / "logistic_model.pkl").open("wb") as f: pickle.dump(artifact,f,pickle.HIGHEST_PROTOCOL)
    report={"method":"official train fit, train calibration; fresh fit-only full-document 160k word TF-IDF; logistic SGD",
            "counts":{"fit":len(fit),"calibration":len(cal),"labels":100},"candidate_reports":reports,
            "selection":best,"elapsed_seconds":round(time.time()-t0,1)}
    (OUT / "logistic_calibration_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("selected",best,"elapsed",round(time.time()-t0,1),flush=True)


if __name__=="__main__": main()
