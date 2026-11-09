import argparse
import json
import re
import sys
from dataclasses import asdict,replace
from pathlib import Path

from .contracts import CorpusConfig,GenerationConfig,ModelConfig,TrainConfig
from .io import write_json_atomic,sha256_file
from .prepare import load_prepared,prepare_data
from .submission import extract_official_test_prompts,load_json_strict,validate_predictions,write_submission

DEFAULT_CONFIG = Path(__file__).parents[1]/"configs/a.json"
OFFICIAL_TEST_REPO = "Exploration-Lab/CS-781-Capstone"
OFFICIAL_TEST_FILE = "Phase_1_Pretraining/Test/test.json"


def _prepare(args):
    from .corpus import iter_official_documents
    from .splits import catalog_documents,catalog_metadata
    from .tokenizer import fit_tokenizer,validate_tokenizer_provenance
    root=Path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    cfg=CorpusConfig(revision=args.revision,seed=args.seed,max_chars_per_source=args.max_chars_per_source)
    database=root/"catalog.sqlite"
    if database.exists():
        if catalog_metadata(database)["corpus_config"]!=asdict(cfg):
            raise ValueError("Existing catalog uses different acquisition settings")
    else:
        catalog_documents(iter_official_documents(cfg),database,args.seed,corpus_config=cfg)
    tokenizer=root/"tokenizer.json"
    if tokenizer.exists():
        validate_tokenizer_provenance(tokenizer,database)
    else:
        fit_tokenizer(database,tokenizer,32768,args.tokenizer_chars)
    output=root/"prepared"
    data=load_prepared(output) if output.exists() else prepare_data(database,tokenizer,output,args.seed,max_seq_len=args.context)
    print(json.dumps({"data":str(data.root),"manifest_sha256":data.manifest_sha256,"counts":[data.train_examples,data.dev_examples,data.confirmation_examples]}))


def _train(args,benchmark=False):
    from .train import train_a
    config=load_json_strict(args.config)
    model=ModelConfig(**config["model"])
    train=TrainConfig(target_tokens=args.target_tokens,**config["train"])
    if args.device:
        train=replace(train,device=args.device)
    if benchmark:
        train=replace(train,target_tokens=10**15,warmup_ratio=0,log_every=1,save_every=125)
    state=train_a(load_prepared(args.data),model,train,Path(args.run_dir),args.resume,
                  stop_after_steps=125 if benchmark else args.stop_after_steps,smoke_test=args.smoke_test)
    if benchmark:
        with (Path(args.run_dir)/"metrics.jsonl").open() as handle:
            rows=[json.loads(line) for line in handle]
        first,last=rows[24],rows[124]
        seconds=(last["gpu_hours_this_session"]-first["gpu_hours_this_session"])*3600
        if seconds<=0:
            raise ValueError("GPU benchmark needs CUDA and 125 completed steps")
        write_json_atomic(Path(args.run_dir)/"benchmark.json",{"warmup_steps":25,"measured_steps":100,
                         "seconds":seconds,"input_tokens_per_second":(last["input_tokens"]-first["input_tokens"])/seconds,
                         "target_tokens_per_second":(last["target_tokens"]-first["target_tokens"])/seconds,
                         "peak_memory_bytes":last["peak_memory_bytes"],"state":asdict(state)})
    print(json.dumps(asdict(state)))


def _evaluate(args):
    import torch
    from .evaluate import score_predictions
    from .generate import generate_continuation
    from .models.a import LegalGPT
    from .tokenizer import load_tokenizer
    data=load_prepared(args.data)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    metadata=payload["metadata"]
    if metadata["data_manifest_sha256"]!=data.manifest_sha256 or metadata["tokenizer_sha256"]!=data.tokenizer_sha256:
        raise ValueError("Checkpoint and evaluation data mismatch")
    device=torch.device(args.device or "cuda")
    model=LegalGPT(ModelConfig(**metadata["model_config"])).to(device)
    model.load_state_dict(payload["model"],strict=True)
    tokenizer=load_tokenizer(data.root/"tokenizer.json")
    prompts=load_json_strict(data.root/f"{args.split}.prompts.json")
    refs=load_json_strict(data.root/f"{args.split}.references.json")
    keys=list(prompts)[:args.limit] if args.limit else list(prompts)
    if not keys:
        raise ValueError("No evaluation IDs")
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config=GenerationConfig(**load_json_strict(args.config)["decoding"][args.policy])
    predictions={}
    failures={}
    for key in keys:
        try:
            predictions[key]=generate_continuation(model,tokenizer,prompts[key]["text"],config)
        except (RuntimeError,ValueError) as exc:
            failures[key]=type(exc).__name__
    write_json_atomic(output.with_suffix(".failures.json"),failures)
    if failures:
        raise ValueError(f"Generation failed for {len(failures)} declared IDs; do not score a filtered subset")
    validate_predictions(predictions,{k:prompts[k]["text"] for k in keys})
    write_json_atomic(output,{k:{"text":v} for k,v in predictions.items()})
    reference_path=output.with_suffix(".references.json")
    write_json_atomic(reference_path,{k:refs[k] for k in keys})
    metrics=score_predictions(reference_path,output,Path(__file__).parents[1]/"vendor/phase1_evaluate.py",args.scorer_cache)
    sources={}
    for key in keys:
        sources.setdefault(key.split(":",1)[0],[]).append(key)
    per_source={}
    for source,ids in sources.items():
        source_output=output.with_name(output.stem+"."+source+".json")
        source_ref=source_output.with_suffix(".references.json")
        write_json_atomic(source_output,{k:{"text":predictions[k]} for k in ids})
        write_json_atomic(source_ref,{k:refs[k] for k in ids})
        per_source[source]=score_predictions(source_ref,source_output,Path(__file__).parents[1]/"vendor/phase1_evaluate.py",args.scorer_cache)
    repetitions={k:1-len(set(v.split()))/128 for k,v in predictions.items()}
    report={"checkpoint_sha256":sha256_file(args.checkpoint),"checkpoint_state":payload["state"],"decoding":asdict(config),
            "split":args.split,"limited_sample":bool(args.limit),"synthetic_smoke":metadata["synthetic_smoke"],
            "metrics":metrics,"per_source":per_source,"repeated_word_fraction":repetitions,"declared_ids":keys}
    write_json_atomic(output.with_suffix(".context.json"),report)
    print(json.dumps(report))


def _checkpoint_run_directory(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    latest = checkpoint.parent.name=="checkpoints" and checkpoint.name=="latest.pt"
    candidate = checkpoint.parent.name=="candidates" and re.fullmatch(r"step_\d{7,}\.pt",checkpoint.name) is not None
    if not (latest or candidate):
        raise ValueError("Use latest.pt or a step candidate from a project run directory")
    run_dir = checkpoint.parent.parent
    run_manifest = run_dir/"run.json"
    if not run_manifest.is_file():
        raise ValueError("Checkpoint has no project run.json provenance")
    metadata = load_json_strict(run_manifest)
    if not isinstance(metadata,dict):
        raise ValueError("Project run.json must contain an object")
    return checkpoint,run_dir,metadata


def _official_dataset_revision(manifest):
    catalog = manifest.get("catalog")
    corpus = catalog.get("corpus_config") if isinstance(catalog,dict) else None
    if not isinstance(corpus,dict) or corpus.get("repo_id")!=OFFICIAL_TEST_REPO:
        raise ValueError("Prepared data is not from the pinned official CS781 repository")
    revision = corpus.get("revision")
    if not isinstance(revision,str) or re.fullmatch(r"[0-9a-f]{40}",revision) is None:
        raise ValueError("Prepared data does not record a pinned dataset revision")
    return revision


def _verify_official_checkpoint_provenance(metadata,run_metadata,manifest,data):
    if not isinstance(metadata,dict) or metadata!=run_metadata:
        raise ValueError("Checkpoint metadata does not match its project run.json")
    if metadata.get("synthetic_smoke") is not False or manifest.get("synthetic") is not False:
        raise ValueError("Official-test prediction requires a substantive, non-synthetic run")
    if metadata.get("data_manifest_sha256")!=data.manifest_sha256 or metadata.get("tokenizer_sha256")!=data.tokenizer_sha256:
        raise ValueError("Checkpoint and prepared data/tokenizer provenance differ")
    if re.fullmatch(r"[0-9a-f]{64}",str(metadata.get("code_sha256",""))) is None:
        raise ValueError("Checkpoint run is missing its source-code digest")
    parameter_count = metadata.get("unique_parameters")
    if type(parameter_count) is not int or not 50_000_000<=parameter_count<=300_000_000:
        raise ValueError("Official-test prediction requires a 50M–300M from-scratch model")
    try:
        model_config = ModelConfig(**metadata["model_config"])
        model_config.validate()
    except (KeyError,TypeError,ValueError):
        raise ValueError("Checkpoint model configuration is invalid") from None
    if model_config.max_seq_len!=manifest.get("max_seq_len"):
        raise ValueError("Checkpoint context differs from prepared-data context")
    return model_config,_official_dataset_revision(manifest)


def _download_official_test(revision,cache_dir):
    from huggingface_hub import hf_hub_download
    try:
        path=hf_hub_download(repo_id=OFFICIAL_TEST_REPO,filename=OFFICIAL_TEST_FILE,
                             repo_type="dataset",revision=revision,cache_dir=str(cache_dir))
    except Exception:
        raise RuntimeError("Could not fetch the pinned official test file; check the approved network route and cache") from None
    return Path(path)


def _predict_official_test(args):
    import torch
    from .generate import generate_continuation
    from .models.a import LegalGPT
    from .tokenizer import load_tokenizer

    output=Path(args.output)
    if output.suffix.lower()!=".zip":
        raise ValueError("Official predictions must be packaged to a .zip output")
    predictions_path=output.with_name(output.stem+".predictions.json")
    context_path=output.with_name(output.stem+".context.json")
    failures_path=output.with_name(output.stem+".failures.json")
    for path in (output,predictions_path,context_path,failures_path):
        if path.exists():
            raise FileExistsError(path)

    data=load_prepared(args.data)
    manifest=load_json_strict(data.root/"manifest.json")
    checkpoint,run_dir,run_metadata=_checkpoint_run_directory(args.checkpoint)
    # This path is restricted to our own run layout and matching run.json before pickle loading.
    try:
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    except Exception:
        raise ValueError("Trusted project checkpoint could not be loaded") from None
    if not isinstance(payload,dict):
        raise ValueError("Project checkpoint payload must be an object")
    metadata=payload.get("metadata")
    model_config,revision=_verify_official_checkpoint_provenance(metadata,run_metadata,manifest,data)
    if not isinstance(payload.get("model"),dict) or not isinstance(payload.get("state"),dict):
        raise ValueError("Project checkpoint is missing model weights or training state")

    decode=load_json_strict(args.config).get("decoding")
    if not isinstance(decode,list) or args.policy>=len(decode):
        raise ValueError("Selected decoding policy is missing from the config")
    try:
        generation_config=GenerationConfig(**decode[args.policy])
    except (TypeError,ValueError):
        raise ValueError("Selected decoding policy is invalid") from None
    if generation_config.target_words!=128:
        raise ValueError("Official continuation length must remain exactly 128 words")

    cache_dir=Path(args.cache_dir) if args.cache_dir else data.root.parent/"hf_hub"
    test_path=_download_official_test(revision,cache_dir)
    test_records=load_json_strict(test_path)
    prompts=extract_official_test_prompts(test_records)
    device=torch.device(args.device or "cuda")
    model=LegalGPT(model_config).to(device)
    if model.count_parameters()!=metadata["unique_parameters"]:
        raise ValueError("Instantiated model parameter count differs from checkpoint provenance")
    model.load_state_dict(payload["model"],strict=True)
    tokenizer=load_tokenizer(data.root/"tokenizer.json")

    predictions,failures={},{}
    for key,prompt in prompts.items():
        try:
            predictions[key]=generate_continuation(model,tokenizer,prompt,generation_config)
        except (RuntimeError,ValueError) as exc:
            failures[key]=type(exc).__name__
    if failures:
        write_json_atomic(failures_path,{"dataset_revision":revision,"test_sha256":sha256_file(test_path),"failures":failures})
        raise ValueError(f"Generation failed for {len(failures)} official IDs; no partial submission was packaged")

    validate_predictions(predictions,prompts)
    records={key:{"text":text} for key,text in predictions.items()}
    write_json_atomic(predictions_path,records)
    zip_sha256=write_submission(predictions,prompts,output)
    report={"checkpoint_sha256":sha256_file(checkpoint),"checkpoint_state":payload["state"],
            "checkpoint_run_dir":str(run_dir),"code_sha256":metadata["code_sha256"],
            "data_manifest_sha256":data.manifest_sha256,"tokenizer_sha256":data.tokenizer_sha256,
            "dataset_revision":revision,"test_sha256":sha256_file(test_path),
            "generation_config":asdict(generation_config),"synthetic_smoke":False,
            "prediction_count":len(predictions),"prediction_zip_sha256":zip_sha256}
    write_json_atomic(context_path,report)
    print(json.dumps(report,allow_nan=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description="CS781 from-scratch legal continuation pipeline")
    sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--root",required=True)
    prep.add_argument("--revision",required=True)
    prep.add_argument("--max-chars-per-source",type=int,default=100_000_000)
    prep.add_argument("--tokenizer-chars",type=int,default=100_000_000)
    prep.add_argument("--seed",type=int,default=781)
    prep.add_argument("--context",type=int,default=2048)
    validate=sub.add_parser("validate-data")
    validate.add_argument("--data",required=True)
    for name in ("train-a","benchmark"):
        train=sub.add_parser(name)
        train.add_argument("--data",required=True)
        train.add_argument("--run-dir",required=True)
        train.add_argument("--target-tokens",type=int,required=True)
        train.add_argument("--config",default=str(DEFAULT_CONFIG))
        train.add_argument("--device")
        train.add_argument("--resume",action="store_true")
        train.add_argument("--smoke-test",action="store_true")
        train.add_argument("--stop-after-steps",type=int)
    evaluate=sub.add_parser("evaluate")
    for name in ("data","checkpoint","output"):
        evaluate.add_argument("--"+name,required=True)
    evaluate.add_argument("--config",default=str(DEFAULT_CONFIG))
    evaluate.add_argument("--split",choices=["dev","confirmation"],default="dev")
    evaluate.add_argument("--policy",choices=[0,1,2],type=int,default=0)
    evaluate.add_argument("--device")
    evaluate.add_argument("--limit",type=int)
    evaluate.add_argument("--scorer-cache",required=True)
    official=sub.add_parser("predict-official-test")
    for name in ("data","checkpoint","output"):
        official.add_argument("--"+name,required=True)
    official.add_argument("--config",default=str(DEFAULT_CONFIG))
    official.add_argument("--policy",choices=[0,1,2],type=int,default=0)
    official.add_argument("--device")
    official.add_argument("--cache-dir")
    package=sub.add_parser("submission")
    for name in ("predictions","prompts","output"):
        package.add_argument("--"+name,required=True)
    status=sub.add_parser("status")
    status.add_argument("--run-dir",required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="prepare":
            _prepare(args)
        elif args.command=="validate-data":
            print(json.dumps(asdict(load_prepared(args.data)),default=str))
        elif args.command in ("train-a","benchmark"):
            _train(args,args.command=="benchmark")
        elif args.command=="evaluate":
            _evaluate(args)
        elif args.command=="predict-official-test":
            _predict_official_test(args)
        elif args.command=="submission":
            records=load_json_strict(args.predictions)
            prompts=load_json_strict(args.prompts)
            print(write_submission({k:v["text"] for k,v in records.items()},{k:v["text"] for k,v in prompts.items()},args.output))
        elif args.command=="status":
            print(json.dumps(load_json_strict(Path(args.run_dir)/"run.json")))
        return 0
    except (RuntimeError,ValueError,OSError,KeyError,IndexError) as exc:
        print(f"{args.command} failed ({type(exc).__name__}); inspect completed manifests and run artifacts",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
