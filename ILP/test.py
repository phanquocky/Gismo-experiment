"""
Direct test for ZerkaniAdj d=2, l=2 (39 nodes, 493506 constraints).
Runs WITHOUT multiprocessing so the real error is visible instead of
"WORKER CRASHED: A process in the process pool was terminated abruptly".
"""

import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "baseline" / "tools"))

GRAPH_FILE = str(_HERE.parent / "datasets" / "ZerkaniAdj.txt")
D, L = 2, 2


def memory_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1.0


def main():
    print(f"Graph : {GRAPH_FILE}")
    print(f"d={D}  l={L}")
    print("-" * 50)

    from dldisjunct import ilp_dl_disjunct

    mem_before = memory_mb()
    print(f"Memory before solve : {mem_before:.1f} MB")

    try:
        sensor_nodes, obj, elapsed = ilp_dl_disjunct(GRAPH_FILE, D, L)
        mem_after = memory_mb()
        print(f"Memory after solve  : {mem_after:.1f} MB  (delta {mem_after - mem_before:+.1f} MB)")
        print(f"Result  : size={int(obj)}  time={elapsed:.3f}s")
        print(f"Sensors : {sensor_nodes}")

    except SystemExit as e:
        print(f"\n[CRASH] SystemExit raised with code: {e.code}")
        traceback.print_exc()

    except MemoryError as e:
        mem_after = memory_mb()
        print(f"\n[CRASH] MemoryError — RSS at crash: {mem_after:.1f} MB")
        traceback.print_exc()

    except Exception as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
