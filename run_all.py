#!/usr/bin/env python3
"""Alle Schritte am Stueck. Nutzung: python run_all.py [--synth] [--force]"""
import argparse
from src import steps as S
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true", help="synthetische Testdaten (offline)")
    ap.add_argument("--force", action="store_true", help="Cache ignorieren")
    a = ap.parse_args()
    S.run_all(synth=a.synth, force=a.force)
