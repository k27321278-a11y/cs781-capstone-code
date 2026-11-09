"""Full-size engineering profile, not a trained quality experiment."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from legalspan.model import LegalSpanGPT, SpanConfig
from cs781.io import write_json_atomic

p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
torch.set_num_threads(2); torch.manual_seed(781)
cfg = SpanConfig()
model = LegalSpanGPT(cfg).cuda().train()
x = torch.randint(4, cfg.vocab_size, (1, 8192), device='cuda'); x[0, 0] = 1
y = x.roll(-1, 1); y[:, :4095] = -100; y[0, -1] = 2
roles = torch.randint(0, 5, (1, 2048), device='cuda')
y[:, :4095] = x.roll(-1, 1)[:, :4095]; y[:, :2047] = -100
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
torch.cuda.reset_peak_memory_stats(); begin = time.monotonic()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.autocast('cuda', dtype=torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        _, loss, _ = model(x, targets=y, prompt_length=2048, prompt_roles=roles, return_probs=False)
    loss.backward()
optimizer.step()
torch.cuda.synchronize()
assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
events = sorted({e.key for e in prof.key_averages() if 'flash' in e.key.lower()})
assert any('flash_attention' in name for name in events), 'No actual Flash dispatch captured'
result = {'engineering_only': True, 'parameters': model.count_parameters(), 'length': 8192,
          'prompt_length': 2048, 'optimizer_step': True, 'finite_loss': float(loss), 'seconds': time.monotonic()-begin,
          'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'flash_events': events,
          'torch': str(torch.__version__), 'device': torch.cuda.get_device_name(0)}
write_json_atomic(a.output, result); print(json.dumps(result), flush=True)
