"""DinkelbachALNS orchestrator: adaptive large-neighborhood search with
move evaluation via the Dinkelbach parametric function h(x) = units − λ·|A'|.

Loop:
  destroy → repair → accept (SA on Δh) → score → adapt weights.
  Periodically refresh λ from best_sol.ratio().
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from problems.base import ProblemInput

from .operators import (
    d_aisle_based,
    d_density_outlier,
    d_random_order,
    d_shaw,
    d_worst_order,
    r_aisle_aware,
    r_greedy_ratio,
    r_random,
    r_regret2,
)
from .preprocess import Preprocessed
from .state import WaveSolution


@dataclass
class ALNSParams:
    gamma_min: float = 0.10
    gamma_max: float = 0.40
    T_start: float = 1.0
    T_end: float = 0.001
    cooling: float = 0.9995
    sigma1: float = 33.0
    sigma2: float = 9.0
    sigma3: float = 3.0
    decay: float = 0.8
    lam_update_interval: int = 50
    flush_interval: int = 100
    # Check the deadline every iteration: on large instances a single
    # destroy/repair can take >1s, so a coarser interval would overshoot the
    # SIGALRM budget by tens of seconds.
    time_check_interval: int = 1


class DinkelbachALNS:
    def __init__(
        self,
        instance: ProblemInput,
        pre: Preprocessed,
        params: ALNSParams | None = None,
        seed: int = 42,
    ):
        self.instance = instance
        self.pre = pre
        self.p = params or ALNSParams()
        self.rng = random.Random(seed)

        # 5 destroy + 4 repair (closures over instance/pre/rng).
        self.destroy_ops = [
            ("random_order",     self._make_d_random_order()),
            ("worst_order",      self._make_d_worst_order()),
            ("aisle_based",      self._make_d_aisle_based()),
            ("shaw",             self._make_d_shaw()),
            ("density_outlier",  self._make_d_density_outlier()),
        ]
        self.repair_ops = [
            ("greedy_ratio",  self._make_r_greedy_ratio()),
            ("regret2",       self._make_r_regret2()),
            ("aisle_aware",   self._make_r_aisle_aware()),
            ("random",        self._make_r_random()),
        ]
        nd = len(self.destroy_ops)
        nr = len(self.repair_ops)
        self.destroy_w = [1.0] * nd
        self.repair_w = [1.0] * nr
        self.destroy_score = [0.0] * nd
        self.repair_score = [0.0] * nr
        self.destroy_use = [0] * nd
        self.repair_use = [0] * nr

        self.lam = 0.0
        self.T = self.p.T_start
        self.best_sol: WaveSolution | None = None
        self.current_sol: WaveSolution | None = None
        self.iterations = 0

    # --- operator factories (close over rng/instance/pre and sample γ) --- #

    def _gamma(self) -> float:
        return self.rng.uniform(self.p.gamma_min, self.p.gamma_max)

    def _make_d_random_order(self):
        def fn(sol):
            d_random_order(sol, self.rng, self.instance, self._gamma())
        return fn

    def _make_d_worst_order(self):
        def fn(sol):
            d_worst_order(sol, self.rng, self.instance, self.pre, self._gamma())
        return fn

    def _make_d_aisle_based(self):
        def fn(sol):
            d_aisle_based(sol, self.rng, self.instance)
        return fn

    def _make_d_shaw(self):
        def fn(sol):
            d_shaw(sol, self.rng, self.instance, self.pre, self._gamma())
        return fn

    def _make_d_density_outlier(self):
        def fn(sol):
            d_density_outlier(sol, self.rng, self.instance, self.pre, self._gamma())
        return fn

    def _make_r_greedy_ratio(self):
        def fn(sol, lam):
            r_greedy_ratio(sol, lam, self.rng, self.instance, self.pre)
        return fn

    def _make_r_regret2(self):
        def fn(sol, lam):
            r_regret2(sol, lam, self.rng, self.instance, self.pre)
        return fn

    def _make_r_aisle_aware(self):
        def fn(sol, lam):
            r_aisle_aware(sol, lam, self.rng, self.instance, self.pre)
        return fn

    def _make_r_random(self):
        def fn(sol, lam):
            r_random(sol, lam, self.rng, self.instance, self.pre)
        return fn

    # --- adaptive selection ----------------------------------------------- #

    def select(self, weights: list[float]) -> int:
        total = sum(weights)
        if total <= 0:
            return self.rng.randrange(len(weights))
        r = self.rng.uniform(0.0, total)
        cum = 0.0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1

    def update_score(self, d_idx: int, r_idx: int, score: float) -> None:
        self.destroy_score[d_idx] += score
        self.repair_score[r_idx] += score
        self.destroy_use[d_idx] += 1
        self.repair_use[r_idx] += 1

    def flush_weights(self) -> None:
        decay = self.p.decay
        for i in range(len(self.destroy_w)):
            if self.destroy_use[i] > 0:
                avg = self.destroy_score[i] / self.destroy_use[i]
                self.destroy_w[i] = decay * self.destroy_w[i] + (1 - decay) * avg
            self.destroy_score[i] = 0.0
            self.destroy_use[i] = 0
        for i in range(len(self.repair_w)):
            if self.repair_use[i] > 0:
                avg = self.repair_score[i] / self.repair_use[i]
                self.repair_w[i] = decay * self.repair_w[i] + (1 - decay) * avg
            self.repair_score[i] = 0.0
            self.repair_use[i] = 0
        # Floor weights so no operator dies entirely.
        self.destroy_w = [max(w, 0.05) for w in self.destroy_w]
        self.repair_w = [max(w, 0.05) for w in self.repair_w]

    # --- SA acceptance ---------------------------------------------------- #

    def accept(self, new_sol: WaveSolution, current_sol: WaveSolution) -> bool:
        delta = new_sol.dinkelbach_value(self.lam) - current_sol.dinkelbach_value(self.lam)
        if delta >= 0:
            return True
        if self.T <= 0:
            return False
        return self.rng.random() < math.exp(delta / self.T)

    def update_lam(self) -> bool:
        if self.best_sol is None or not self.best_sol.is_feasible(self.instance):
            return False
        new_lam = self.best_sol.ratio()
        if abs(new_lam - self.lam) > 1e-6:
            self.lam = new_lam
            return True
        return False

    # --- main loop -------------------------------------------------------- #

    def run(
        self,
        initial_sol: WaveSolution,
        time_budget: float,
        log_fn=None,
        on_improve=None,
    ) -> WaveSolution:
        """Returns the best feasible solution found within time_budget seconds.
        log_fn(iteration, elapsed, best_ratio, lam, T) is called periodically
        if provided. on_improve(best_sol) is called every time best_sol is
        replaced — the wrapping algorithm uses it to mirror the incumbent into
        last_best so a SIGALRM-interrupted run still emits a feasible record."""
        self.current_sol = initial_sol.copy()
        self.best_sol = initial_sol.copy()
        self.lam = initial_sol.ratio() if initial_sol.is_feasible(self.instance) else 0.0

        t_start = time.time()
        deadline = t_start + time_budget
        last_time_check = t_start
        it = 0

        while True:
            it += 1
            self.iterations = it
            if it % self.p.time_check_interval == 0:
                now = time.time()
                if now > deadline:
                    break
                last_time_check = now

            d_idx = self.select(self.destroy_w)
            r_idx = self.select(self.repair_w)

            trial = self.current_sol.copy()
            self.destroy_ops[d_idx][1](trial)
            self.repair_ops[r_idx][1](trial, self.lam)

            score = 0.0
            if trial.is_feasible(self.instance):
                if trial.ratio() > self.best_sol.ratio() + 1e-9:
                    self.best_sol = trial.copy()
                    self.current_sol = trial
                    score = self.p.sigma1
                    if on_improve is not None:
                        on_improve(self.best_sol)
                elif trial.dinkelbach_value(self.lam) > self.current_sol.dinkelbach_value(self.lam) + 1e-9:
                    self.current_sol = trial
                    score = self.p.sigma2
                elif self.accept(trial, self.current_sol):
                    self.current_sol = trial
                    score = self.p.sigma3

            self.update_score(d_idx, r_idx, score)

            if it % self.p.flush_interval == 0:
                self.flush_weights()

            if it % self.p.lam_update_interval == 0:
                if self.update_lam():
                    # Lukewarm restart of T after λ moves.
                    self.T = max(self.T, self.p.T_start * 0.5)

            self.T = max(self.T * self.p.cooling, self.p.T_end)

            if log_fn and it % 200 == 0:
                log_fn(it, time.time() - t_start, self.best_sol.ratio(), self.lam, self.T)

        return self.best_sol
