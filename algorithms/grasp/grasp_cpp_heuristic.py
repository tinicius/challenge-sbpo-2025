"""GRASP heuristic backed by a native C++ solver for speed.

Same semantics and CLI parameters as ``GraspHeuristic``; the actual search
runs in a subprocess. Progress lines are streamed on stdout so ``last_best``
is always up to date, even if the Python-side SIGALRM kills the solve.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from algorithms.base import Algorithm
from problems.base import ProblemInput


_VALID_CONSTRUCTION = {"size", "synergy", "aisle_cost", "aisle_cost_fast"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_LOCAL_SEARCH = {"none", "swap", "full"}

_BINARY_DIR = Path(__file__).resolve().parent / "cpp"
_BINARY_PATH = _BINARY_DIR / "grasp_solver"
_BUILD_SCRIPT = _BINARY_DIR / "build.sh"

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


def _ensure_binary() -> Path:
    src = _BINARY_DIR / "grasp_solver.cpp"
    if _BINARY_PATH.exists() and _BINARY_PATH.stat().st_mtime >= src.stat().st_mtime:
        return _BINARY_PATH
    if not _BUILD_SCRIPT.exists():
        raise RuntimeError(f"missing build script at {_BUILD_SCRIPT}")
    subprocess.run(["bash", str(_BUILD_SCRIPT)], check=True)
    return _BINARY_PATH


def _write_instance(instance: ProblemInput, path: str) -> None:
    parts: list[str] = [f"{instance.nOrders} {instance.nItems} {instance.nAisles}"]
    for order in instance.orders:
        tokens = [str(len(order))]
        for item, qty in order.items():
            tokens.append(str(item))
            tokens.append(str(qty))
        parts.append(" ".join(tokens))
    for aisle in instance.aisles:
        tokens = [str(len(aisle))]
        for item, qty in aisle.items():
            tokens.append(str(item))
            tokens.append(str(qty))
        parts.append(" ".join(tokens))
    parts.append(f"{instance.lb} {instance.ub}")
    with open(path, "w") as f:
        f.write("\n".join(parts))
        f.write("\n")


class GraspCppHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        alpha = params.get("alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(f"GraspCppHeuristic: invalid 'alpha'={alpha!r}")
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"GraspCppHeuristic: 'alpha'={alpha} out of [0,1]")

        construction = params.get("construction_score")
        if construction not in _VALID_CONSTRUCTION:
            raise ValueError(
                f"GraspCppHeuristic: invalid 'construction_score'={construction!r}"
            )

        max_iterations = params.get("max_iterations")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"GraspCppHeuristic: invalid 'max_iterations'={max_iterations!r}"
            )

        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(f"GraspCppHeuristic: invalid 'greedy'={greedy!r}")

        local_search = params.get("local_search")
        if local_search not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"GraspCppHeuristic: invalid 'local_search'={local_search!r}"
            )

        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                f"GraspCppHeuristic: invalid 'similarity_weighted'={similarity_weighted!r}"
            )

        time_limit = params.get("time_limit")
        if time_limit is not None and (
            not isinstance(time_limit, (int, float))
            or isinstance(time_limit, bool)
            or time_limit <= 0
        ):
            raise ValueError(f"GraspCppHeuristic: invalid 'time_limit'={time_limit!r}")

        self._alpha = alpha
        self._construction = construction
        self._max_iterations = max_iterations
        self._greedy = greedy
        self._local_search = local_search
        self._similarity_weighted = similarity_weighted
        self._seed = params.get("seed")
        self._time_limit = time_limit

    @property
    def name(self) -> str:
        return "grasp_cpp_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)
        if instance.nOrders == 0 or instance.nAisles == 0:
            return dict(_EMPTY_RESULT)

        binary = _ensure_binary()

        with tempfile.TemporaryDirectory(prefix="grasp_cpp_") as tmp:
            inst_path = os.path.join(tmp, "instance.txt")
            _write_instance(instance, inst_path)

            args = [
                str(binary),
                inst_path,
                f"--alpha={self._alpha}",
                f"--construction={self._construction}",
                f"--max-iterations={self._max_iterations}",
                f"--greedy={self._greedy}",
                f"--local-search={self._local_search}",
                f"--similarity-weighted={1 if self._similarity_weighted else 0}",
            ]
            if self._seed is not None:
                args.append(f"--seed={self._seed}")
            if self._time_limit is not None:
                args.append(f"--time-limit={self._time_limit}")

            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    obj = record.get("objective", 0.0)
                    if obj > self.last_best["objective"]:
                        self.last_best = {
                            "selected_orders": record.get("selected_orders", []),
                            "visited_aisles": record.get("visited_aisles", []),
                            "objective": float(obj),
                        }
                proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise

            if proc.returncode != 0 and self.last_best["objective"] == 0.0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"grasp_solver exited {proc.returncode}: {stderr.strip()}"
                )

        return {
            "selected_orders": self.last_best["selected_orders"],
            "visited_aisles": self.last_best["visited_aisles"],
            "objective": self.last_best["objective"],
        }
