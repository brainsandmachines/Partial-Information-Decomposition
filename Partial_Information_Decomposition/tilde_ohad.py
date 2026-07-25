

import os
from pathlib import Path
import sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))
wrapper_root = root / "library_wrappers"
if str(wrapper_root) not in sys.path:
    sys.path.insert(0, str(wrapper_root))

from external.gpid.src.gpid.tilde_pid import exact_tilde_union_info_minimizer,objective





def calculate_tilde_union_info(hx, hy, reg=1e-7, max_iters=20000, verbose=False):
    pass

