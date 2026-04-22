"""Simple heuristic backed by a native C++ solver.

Matches SimpleHeuristic parameters and output contract while delegating the
compute-heavy routine to a subprocess.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from algorithms.base import Algorithm
from problems.base import ProblemInput

_VALID_ORDER = {None, "asc", "desc", "similar", "diff"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_FIRST_ORDER = {None, "smaller", "bigger"}

_BINARY_DIR = Path(__file__).resolve().parent / "cpp"
_BINARY_PATH = _BINARY_DIR / "simple_solver"
_BUILD_SCRIPT = _BINARY_DIR / "build.sh"

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


def _ensure_binary() -> Path:
    src = _BINARY_DIR / "simple_solver.cpp"
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


class SimpleCppHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        order = params.get("order")
        if order not in _VALID_ORDER:
            raise ValueError(
                f"SimpleCppHeuristic: invalid 'order'={order!r}; "
                f"expected one of {sorted(v for v in _VALID_ORDER if v)} or unset"
            )

        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"SimpleCppHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )

        first_order = params.get("first_order")
        if first_order not in _VALID_FIRST_ORDER:
            raise ValueError(
                f"SimpleCppHeuristic: invalid 'first_order'={first_order!r}; "
                f"expected one of {sorted(v for v in _VALID_FIRST_ORDER if v)} or unset"
            )

        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                "SimpleCppHeuristic: invalid 'similarity_weighted'="
                f"{similarity_weighted!r}; expected bool"
            )

        seed = params.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(
                f"SimpleCppHeuristic: invalid 'seed'={seed!r}; expected int or unset"
            )

        self._order = order
        self._greedy = greedy
        self._first_order = first_order
        self._similarity_weighted = similarity_weighted
        self._seed = seed

    @property
    def name(self) -> str:
        return "simple_cpp_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)
        if instance.nOrders == 0 or instance.nAisles == 0:
            return dict(_EMPTY_RESULT)

        binary = _ensure_binary()

        with tempfile.TemporaryDirectory(prefix="simple_cpp_") as tmp:
            inst_path = os.path.join(tmp, "instance.txt")
            _write_instance(instance, inst_path)

            args = [
                str(binary),
                inst_path,
                f"--greedy={self._greedy}",
                f"--similarity-weighted={1 if self._similarity_weighted else 0}",
            ]
            if self._order is not None:
                args.append(f"--order={self._order}")
            if self._first_order is not None:
                args.append(f"--first-order={self._first_order}")
            if self._seed is not None:
                args.append(f"--seed={self._seed}")

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

            if proc.returncode != 0:
                if self.last_best["objective"] > 0.0:
                    return dict(self.last_best)
                return dict(_EMPTY_RESULT)

        return {
            "selected_orders": self.last_best["selected_orders"],
            "visited_aisles": self.last_best["visited_aisles"],
            "objective": self.last_best["objective"],
        }
