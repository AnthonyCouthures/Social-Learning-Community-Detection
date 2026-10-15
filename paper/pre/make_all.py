"""Reproduce every figure of the paper.

python paper/pre/make_all.py            # full runs (Figs. 3, 4 and 9 take a while)
python paper/pre/make_all.py --quick    # fast smoke test of the whole pipeline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = [
    "fig01_signal_functions.py",
    "fig02_karate_ser.py",
    "fig03_spectral_thresholds.py",
    "fig04_benchmark_abcd.py",
    "fig05_chimpanzees.py",
    "fig06_football.py",
    "fig07_rnp_steps.py",
    "fig08_rnp_dc_multiplateau.py",
    "fig09_rnp_dc_sensitivity.py",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="fast smoke test")
    p.add_argument("--only", nargs="+", metavar="N", help="figure numbers to build, e.g. --only 2 5")
    args = p.parse_args()
    scripts = [
        s for s in SCRIPTS if not args.only or any(s.startswith(f"fig{int(n):02d}") for n in args.only)
    ]
    for script in scripts:
        start = time.perf_counter()
        print(f"=== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *(["--quick"] if args.quick else [])], check=True)
        print(f"    done in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
