import json
import os
import random
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .contracts import CheckpointMismatch, RunAlreadyActive, TrainState


@contextmanager
def run_lock(run_dir):
    import fcntl
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    with (run_dir/"run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunAlreadyActive("A trainer already holds this run's lock") from None
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def save_checkpoint(path,model,optimizer,state,sampler_state,metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"state":asdict(state),
               "sampler":sampler_state,"metadata":metadata,
               "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
                      "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    with temporary.open("wb") as handle:
        torch.save(payload,handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def restore_checkpoint(path,model,optimizer,expected_metadata):
    # Only our own trusted run artifacts; never load arbitrary downloaded pickle weights.
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("metadata") != expected_metadata:
        raise CheckpointMismatch("Checkpoint model/data/tokenizer/training/code metadata mismatch")
    model.load_state_dict(payload["model"],strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise CheckpointMismatch("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(rng["cuda"])
    return TrainState(**payload["state"]),payload["sampler"]
