"""
Record which correctness gates (PLAN.md 7.3: G1 forward, G2 backward, G3 compile integration,
G4 loss curve) a result row has passed, in its `gates_passed` list.

The results database rows are the per-case JSON files written by cisc662/bench/kbench.py
(results/db/<group>/<case>.json). record_gate is idempotent, keeps every other field, and rewrites
the file atomically (temp file + os.replace) so a killed job never leaves a truncated row.
Plotting/table scripts call has_gates(row) and refuse rows missing G1-G2.

Run:  python -m cisc662.tests.gates cisc662/results/db/<group>/<case>.json G1 [G2 ...]
"""
import argparse
import json
import os
import re
import tempfile

GATES = ("G1", "G2", "G3", "G4")
REQUIRED_FOR_TABLES = ("G1", "G2")            # kernel rows
REQUIRED_FOR_END_TO_END = ("G1", "G2", "G3", "G4")


def _check_gate(gate):
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {GATES}")


def record_gate(result_json_path, gate):
    """Append `gate` to the row's gates_passed (once). Returns the updated list.
    The row must already exist -- a gate is never recorded for a result that was not written."""
    _check_gate(gate)
    with open(result_json_path) as f:
        row = json.load(f)
    gates = row.get("gates_passed") or []
    if gate not in gates:
        gates.append(gate)
    row["gates_passed"] = gates
    d = os.path.dirname(os.path.abspath(result_json_path))
    fd, tmp = tempfile.mkstemp(prefix=".gate_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(row, f, indent=2)
            f.write("\n")
        os.replace(tmp, result_json_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return gates


def has_gates(row, required=REQUIRED_FOR_TABLES):
    """True if the row (dict, or path to its JSON) has passed every gate in `required`."""
    if isinstance(row, (str, os.PathLike)):
        with open(row) as f:
            row = json.load(f)
    return set(required) <= set(row.get("gates_passed") or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result_json")
    ap.add_argument("gates", nargs="+", help="G1..G4")
    args = ap.parse_args()
    for g in args.gates:  # validate all first, so a typo never leaves a half-recorded row
        if not re.fullmatch(r"G[1-4]", g):
            ap.error(f"bad gate {g!r}")
    for g in args.gates:
        gates = record_gate(args.result_json, g)
    print(f"{args.result_json}: gates_passed={gates}")


if __name__ == "__main__":
    main()
