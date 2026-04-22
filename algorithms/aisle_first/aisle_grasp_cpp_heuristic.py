"""Aisle-centric GRASP heuristic backed by a native C++ solver.

Mirrors ``AisleGraspHeuristic`` parameters and output contract while
delegating the search to a subprocess. Progress lines are streamed on
stdout so ``last_best`` stays current even if the Python-side SIGALRM
kills the solve mid-run.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from algorithms.base import Algorithm
from algorithms.utils.aisle_rank import VALID_AISLE_SCORE, VALID_ORDER_MODE
from problems.base import ProblemInput


_VALID_SCORING = {"static", "adaptive"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_LOCAL_SEARCH = {"none", "swap", "full"}

_BINARY_DIR = Path(__file__).resolve().parent / "cpp"
_BINARY_PATH = _BINARY_DIR / "aisle_grasp_solver"
_BUILD_SCRIPT = _BINARY_DIR / "build.sh"

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


def _ensure_binary() -> Path:
    src = _BINARY_DIR / "aisle_grasp_solver.cpp"
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


class AisleGraspCppHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        alpha = params.get("alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'alpha'={alpha!r}; expected float in [0, 1]"
            )
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"AisleGraspCppHeuristic: 'alpha'={alpha} out of range; expected [0, 1]"
            )

        max_iterations = params.get("max_iterations")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'max_iterations'={max_iterations!r}; "
                f"expected positive int"
            )

        scoring = params.get("scoring", "static")
        if scoring not in _VALID_SCORING:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'scoring'={scoring!r}; "
                f"expected one of {sorted(_VALID_SCORING)}"
            )

        aisle_score = params.get("aisle_score", "useful")
        if aisle_score not in VALID_AISLE_SCORE:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'aisle_score'={aisle_score!r}; "
                f"expected one of {sorted(VALID_AISLE_SCORE)}"
            )

        packing_order = params.get("packing_order")
        if packing_order not in VALID_ORDER_MODE:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'packing_order'={packing_order!r}; "
                f"expected one of {sorted(v for v in VALID_ORDER_MODE if v)} or unset"
            )

        greedy = params.get("greedy", "simple")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )

        local_search_aisle = params.get("local_search_aisle", "none")
        if local_search_aisle not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'local_search_aisle'={local_search_aisle!r}; "
                f"expected one of {sorted(_VALID_LOCAL_SEARCH)}"
            )

        local_search_order = params.get("local_search_order", "none")
        if local_search_order not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'local_search_order'={local_search_order!r}; "
                f"expected one of {sorted(_VALID_LOCAL_SEARCH)}"
            )

        time_limit = params.get("time_limit")
        if time_limit is not None and (
            not isinstance(time_limit, (int, float))
            or isinstance(time_limit, bool)
            or time_limit <= 0
        ):
            raise ValueError(
                f"AisleGraspCppHeuristic: invalid 'time_limit'={time_limit!r}"
            )

        self._alpha = alpha
        self._max_iterations = max_iterations
        self._scoring = scoring
        self._aisle_score = aisle_score
        self._packing_order = packing_order
        self._greedy = greedy
        self._local_search_aisle = local_search_aisle
        self._local_search_order = local_search_order
        self._seed = params.get("seed")
        self._time_limit = time_limit

    @property
    def name(self) -> str:
        return "aisle_grasp_cpp_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)
        if instance.nOrders == 0 or instance.nAisles == 0:
            return dict(_EMPTY_RESULT)

        binary = _ensure_binary()

        # Python's `None` packing_order maps to "shuffle" on the CLI.
        cli_packing = self._packing_order if self._packing_order is not None else "shuffle"

        with tempfile.TemporaryDirectory(prefix="aisle_grasp_cpp_") as tmp:
            inst_path = os.path.join(tmp, "instance.txt")
            _write_instance(instance, inst_path)

            args = [
                str(binary),
                inst_path,
                f"--alpha={self._alpha}",
                f"--scoring={self._scoring}",
                f"--aisle-score={self._aisle_score}",
                f"--packing-order={cli_packing}",
                f"--max-iterations={self._max_iterations}",
                f"--greedy={self._greedy}",
                f"--local-search-aisle={self._local_search_aisle}",
                f"--local-search-order={self._local_search_order}",
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
                    obj = float(record.get("objective", 0.0))
                    if obj > self.last_best["objective"]:
                        self.last_best = {
                            "selected_orders": record.get("selected_orders", []),
                            "visited_aisles": record.get("visited_aisles", []),
                            "objective": obj,
                        }
                proc.wait()
            except BaseException:
                proc.kill()
                proc.wait()
                raise

            if proc.returncode != 0 and self.last_best["objective"] == 0.0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"aisle_grasp_solver exited {proc.returncode}: {stderr.strip()}"
                )

        return {
            "selected_orders": self.last_best["selected_orders"],
            "visited_aisles": self.last_best["visited_aisles"],
            "objective": self.last_best["objective"],
        }
