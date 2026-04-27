"""
Dinkelbach-ALNS matheuristic for the wave order-picking problem.

Pipeline (4 layers, time-budgeted proportionally to `time_limit`):
  1. Preprocess + data structures   (≤ 2.5 %, cap 15 s)
  2. Constructive heuristics         (≤ 4 %,  cap 25 s)
  3. ALNS with Dinkelbach Δh         (rest of the budget, min 60 % when LB on)
  4. CP-SAT Local Branching refinement (15 %, cap 90 s, min 10 s; optional)
"""

from __future__ import annotations

import random
import time

import numpy as np

from algorithms.base import Algorithm
from problems.base import ProblemInput
from problems.validation import compute_objective, is_solution_feasible

from .alns import ALNSParams, DinkelbachALNS as ALNSEngine
from .constructives import build_initial_candidates
from .local_branching import local_branching_refinement
from .preprocess import preprocess
from .state import WaveSolution


_EMPTY_RESULT: dict = {
    "selected_orders": [],
    "visited_aisles": [],
    "objective": 0.0,
}

# The runner enforces an external SIGALRM at exactly `time_limit`. Reserve a
# safety margin so we return before that fires — otherwise the result is lost
# and `last_best` is the only fallback.
_SIGALRM_SAFETY_S = 1.5


class DinkelbachALNS(Algorithm):

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.last_best: dict = dict(_EMPTY_RESULT)

    @property
    def name(self) -> str:
        return "dinkelbach_alns"

    def _commit(self, instance: ProblemInput, sol: WaveSolution) -> None:
        """Mirror a feasible improving solution into self.last_best."""
        if sol is None or not sol.is_feasible(instance):
            return
        selected = sorted(sol.orders)
        aisles = sorted(sol.aisles)
        if not is_solution_feasible(instance, selected, aisles):
            return
        obj = compute_objective(instance, selected, aisles)
        if obj > self.last_best["objective"]:
            self.last_best = {
                "selected_orders": selected,
                "visited_aisles": aisles,
                "objective": obj,
            }

    def solve(self, instance: ProblemInput) -> dict:
        params = self.params or {}
        external_time_limit = float(params.get("time_limit", 590))
        # Reserve a sliver so we return strictly before the runner's SIGALRM.
        time_limit = max(1.0, external_time_limit - _SIGALRM_SAFETY_S)
        seed = int(params.get("seed", 42))
        alns_cfg = params.get("alns", {}) or {}
        lb_cfg = params.get("local_branching", {}) or {}
        ms_cfg = params.get("multi_start", {}) or {}

        random.seed(seed)
        np.random.seed(seed)

        self.last_best = dict(_EMPTY_RESULT)

        deadline = time.time() + time_limit

        # ----------------- time budget split -------------------- #
        t_pre_cap = min(0.025 * time_limit, 15.0)
        t_ctor_cap = min(0.04 * time_limit, 25.0)
        lb_enabled = bool(lb_cfg.get("enabled", True))
        t_lb_cap = (
            max(min(0.15 * time_limit, 90.0), 10.0) if lb_enabled else 0.0
        )

        # ----------------- 1. preprocess ------------------------ #
        t0 = time.time()
        pre = preprocess(instance, prune_aisles=True)
        t_pre = time.time() - t0
        if t_pre > t_pre_cap * 1.5:
            print(
                f"[dalns] preprocess overran budget: {t_pre:.2f}s "
                f"(cap {t_pre_cap:.1f}s)",
                flush=True,
            )

        # ----------------- 2. constructives --------------------- #
        ms_enabled = bool(ms_cfg.get("enabled", True))
        ms_max = max(1, int(ms_cfg.get("max_starts", 4)))
        ms_min_per_start = float(ms_cfg.get("min_time_per_start", 8.0))

        t0 = time.time()
        candidates = build_initial_candidates(
            instance, pre, max_candidates=ms_max if ms_enabled else 1
        )
        t_ctor = time.time() - t0

        if not candidates:
            return dict(self.last_best)

        # Commit the best constructive immediately — guarantees a feasible
        # result even if every later phase is interrupted by SIGALRM.
        self._commit(instance, candidates[0])

        remaining = deadline - time.time()
        t_lb = min(t_lb_cap, max(0.0, remaining * 0.5)) if lb_enabled else 0.0
        t_alns_total = max(remaining - t_lb, 1.0)

        # Decide how many starts actually run: cap by max_starts AND by what
        # fits in the budget under min_time_per_start. Single-start when tight.
        n_starts = len(candidates)
        if ms_enabled and n_starts > 1 and ms_min_per_start > 0:
            n_starts = min(n_starts, max(1, int(t_alns_total // ms_min_per_start)))
        else:
            n_starts = 1
        starts = candidates[:n_starts]
        t_per_start = t_alns_total / n_starts

        print(
            f"[dalns] t=0.0  pre={t_pre:.2f}s ctor={t_ctor:.2f}s "
            f"starts={n_starts}/{len(candidates)} "
            f"r0={[f'{s.ratio():.2f}' for s in starts]} "
            f"budget alns={t_alns_total:.1f}s ({t_per_start:.1f}s/start) "
            f"lb={t_lb:.1f}s",
            flush=True,
        )

        # ----------------- 3. ALNS (multi-start) ---------------- #
        alns_params = ALNSParams(
            gamma_min=float(alns_cfg.get("gamma_min", 0.10)),
            gamma_max=float(alns_cfg.get("gamma_max", 0.40)),
            T_start=float(alns_cfg.get("T_start", 1.0)),
            T_end=float(alns_cfg.get("T_end", 0.001)),
            cooling=float(alns_cfg.get("cooling", 0.9995)),
            sigma1=float(alns_cfg.get("sigma1", 33.0)),
            sigma2=float(alns_cfg.get("sigma2", 9.0)),
            sigma3=float(alns_cfg.get("sigma3", 3.0)),
            decay=float(alns_cfg.get("decay", 0.8)),
            lam_update_interval=int(alns_cfg.get("lam_update_interval", 50)),
            flush_interval=int(alns_cfg.get("flush_interval", 100)),
            time_check_interval=int(alns_cfg.get("time_check_interval", 25)),
        )

        def _on_improve(sol: WaveSolution) -> None:
            self._commit(instance, sol)

        best: WaveSolution = candidates[0]
        for i, start_sol in enumerate(starts):
            # Recompute remaining each iteration so a fast finish on start i
            # gives extra slack to start i+1 (rather than wasting the budget).
            rem = deadline - time.time() - t_lb
            if rem <= 1.0:
                break
            remaining_starts = n_starts - i
            t_budget = max(1.0, rem / remaining_starts)
            engine = ALNSEngine(instance, pre, alns_params, seed=seed + i)
            t_a0 = time.time()
            local_best = engine.run(
                start_sol, time_budget=t_budget, on_improve=_on_improve
            )
            t_used = time.time() - t_a0
            if local_best.is_feasible(instance) and local_best.ratio() > best.ratio() + 1e-9:
                best = local_best
            self._commit(instance, best)
            print(
                f"[dalns] t={time.time() - (deadline - time_limit):.1f}  "
                f"start {i + 1}/{n_starts} iters={engine.iterations} "
                f"r_local={local_best.ratio():.3f} r_best={best.ratio():.3f} "
                f"(ran {t_used:.1f}s of {t_budget:.1f}s)",
                flush=True,
            )

        # ----------------- 4. Local Branching ------------------- #
        if lb_enabled:
            remaining = deadline - time.time()
            t_lb_actual = min(t_lb_cap, max(0.0, remaining))
            if t_lb_actual >= 5.0 and best.is_feasible(instance):
                k_values = list(lb_cfg.get("k_values", [5, 10, 20]))
                num_workers = int(lb_cfg.get("num_workers", 4))
                t_l0 = time.time()
                refined = local_branching_refinement(
                    best, instance, pre,
                    lam=best.ratio(), time_limit=t_lb_actual,
                    k_values=k_values, num_workers=num_workers, seed=seed,
                )
                if refined.ratio() > best.ratio() + 1e-9:
                    best = refined
                self._commit(instance, best)
                print(
                    f"[dalns] t={time.time() - (deadline - time_limit):.1f}  "
                    f"lb done r_lb={best.ratio():.3f} "
                    f"(ran {time.time() - t_l0:.1f}s)",
                    flush=True,
                )

        # ----------------- finalize ----------------------------- #
        self._commit(instance, best)
        return dict(self.last_best)
