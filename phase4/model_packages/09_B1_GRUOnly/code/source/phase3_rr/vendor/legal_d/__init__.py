from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
for p in (root/'vendor',root/'vendor/backbones'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
