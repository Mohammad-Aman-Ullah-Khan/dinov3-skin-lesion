"""
hso.py
Harmony Search Optimization (Section 3.10 / Table 5), nested inside each outer
fold's development data.

Spec implemented exactly as stated:
  - HMS = 8, HMCR = 0.90, PAR = 0.30, 25 improvisations after HM init
    (33 candidate evaluations per outer fold).
  - Search space: backbone LR [5e-6, 5e-5] and module LR [5e-5, 5e-4] in
    log10 coordinates; weight decay [0.01, 0.10]; embedding dropout
    [0.10, 0.40] in linear coordinates.
  - Pitch adjustment in NORMALIZED [0,1] coordinates with fixed bandwidth
    0.05 of the range, clipped to bounds.
  - Objective: maximize inner-validation macro AUPRC; macro F1 tie-break at
    reporting precision (4 decimals).

WARNING (reviewer flag): 33 candidates x 5 folds x 60 epochs of ConvNeXt-B is
165 full trainings. The manuscript states no reduced proxy budget. Either run
it as written, or pass a shortened per-candidate epoch budget (cfg.hso_epochs)
AND declare that budget in the paper.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass
class SearchVar:
    name: str
    lo: float
    hi: float
    log10: bool = False

    def to_norm(self, value: float) -> float:
        a, b = ((math.log10(self.lo), math.log10(self.hi)) if self.log10
                else (self.lo, self.hi))
        v = math.log10(value) if self.log10 else value
        return (v - a) / (b - a)

    def from_norm(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        a, b = ((math.log10(self.lo), math.log10(self.hi)) if self.log10
                else (self.lo, self.hi))
        v = a + u * (b - a)
        return 10 ** v if self.log10 else v


def default_search_space(cfg) -> List[SearchVar]:
    return [
        SearchVar("lr_backbone", *cfg.hso_lr_backbone_range, log10=True),
        SearchVar("lr_modules", *cfg.hso_lr_modules_range, log10=True),
        SearchVar("weight_decay", *cfg.hso_wd_range, log10=False),
        SearchVar("dropout", *cfg.hso_dropout_range, log10=False),
    ]


def harmony_search(
    evaluate: Callable[[Dict[str, float]], Tuple[float, float]],
    space: List[SearchVar],
    hms: int = 8,
    hmcr: float = 0.90,
    par: float = 0.30,
    bandwidth: float = 0.05,
    iterations: int = 25,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, float], Tuple[float, float], List[dict]]:
    """
    evaluate(params) -> (macro_auprc, macro_f1) on the inner validation set.
    Returns (best_params, best_score_tuple, full_history).
    Score comparison: AUPRC rounded to 4 dp first, then macro F1.
    """
    rng = random.Random(seed)
    key = lambda s: (round(s[0], 4), s[1])

    memory: List[Tuple[List[float], Tuple[float, float]]] = []
    history: List[dict] = []

    def decode(u: List[float]) -> Dict[str, float]:
        return {v.name: v.from_norm(ui) for v, ui in zip(space, u)}

    def run(u: List[float], tag: str):
        params = decode(u)
        score = evaluate(params)
        history.append({"tag": tag, "params": params,
                        "macro_auprc": score[0], "macro_f1": score[1]})
        log(f"[HSO {tag}] {params} -> AUPRC={score[0]:.4f} F1={score[1]:.4f}")
        return score

    # ---- harmony memory initialization (8 random harmonies)
    for i in range(hms):
        u = [rng.random() for _ in space]
        memory.append((u, run(u, f"init {i + 1}/{hms}")))

    # ---- 25 improvisations
    for it in range(iterations):
        new_u = []
        for d in range(len(space)):
            if rng.random() < hmcr:                       # memory consideration
                val = rng.choice(memory)[0][d]
                if rng.random() < par:                    # pitch adjustment
                    val += (rng.random() * 2 - 1) * bandwidth
                new_u.append(min(max(val, 0.0), 1.0))
            else:                                         # random selection
                new_u.append(rng.random())
        score = run(new_u, f"improv {it + 1}/{iterations}")
        worst = min(range(len(memory)), key=lambda i: key(memory[i][1]))
        if key(score) > key(memory[worst][1]):
            memory[worst] = (new_u, score)

    best_u, best_score = max(memory, key=lambda m: key(m[1]))
    return decode(best_u), best_score, history
