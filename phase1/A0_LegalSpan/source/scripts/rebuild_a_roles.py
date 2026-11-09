import argparse
from pathlib import Path
from legalspan.data import rebuild_roles
p=argparse.ArgumentParser(); p.add_argument('--parent',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); rebuild_roles(a.parent,a.output)
