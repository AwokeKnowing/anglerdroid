"""python -m emu.tools.slamlog.baseline <logdir>: metrics of the live kslam pose and of raw wheel+gyro odom recorded by nav."""
import json, sys
import numpy as np
from .log import Log
from .metrics import evaluate
L = Log(sys.argv[1]); N = L.nav(); out = {}
for k in ("pose", "odom"):
    est = np.column_stack([N["t_od"], N[k][:, 0], N[k][:, 1], np.unwrap(N[k][:, 2])])
    out["kslam_live" if k == "pose" else "odom_live"] = evaluate(est, L)
print(json.dumps(out, indent=1))
