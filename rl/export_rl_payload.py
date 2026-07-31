"""Export the DeepSWE routing payload for the box-6 soft-kNN RL runs.

Ships everything explicit so the box script replicates lr-baseline splits exactly:
graded/cost matrices, qids, repo codes, and per-seed (tr, te, inner_tr, inner_val).
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from router.experiments import _lrb_deepswe41, split_by_repo_holdout, sub_holdout  # noqa: E402

m = _lrb_deepswe41()
m.emb = np.zeros((m.n, 1))  # sub_holdout slices emb; splits only need groups
splits = {}
for seed in range(6):
    tr, te = split_by_repo_holdout(m, 0.8, seed)
    TR = sub_holdout(m, tr)
    itr, ite = split_by_repo_holdout(TR, 0.75, 100 + seed)
    splits[str(seed)] = {"tr": tr.tolist(), "te": te.tolist(),
                         "inner_tr": tr[itr].tolist(), "inner_val": tr[ite].tolist()}

out = pathlib.Path("/tmp/router_rl_payload.npz")
np.savez_compressed(out, graded=m.graded.astype(np.float32), cost=m.cost.astype(np.float32))
meta = {"arms": m.arms, "qids": m.qids, "groups": m.group, "splits": splits}
pathlib.Path("/tmp/router_rl_meta.json").write_text(json.dumps(meta))
print(f"{len(m.arms)} arms x {m.n} tasks; splits for seeds 0-5 -> {out} + meta.json")
