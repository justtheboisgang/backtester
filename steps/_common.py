"""Gemeinsamer Bootstrap fuer die nummerierten Schritt-Skripte."""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import steps as S  # noqa: E402

def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true",
                    help="synthetische Testdaten statt echtem Download (offline)")
    ap.add_argument("--force", action="store_true",
                    help="Cache ignorieren und neu berechnen")
    return ap.parse_args()
