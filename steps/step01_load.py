#!/usr/bin/env python3
"""Schritt 1: Rohdaten laden. Nutzung: python steps/step01_load.py [--synth] [--force]"""
from _common import S, args
if __name__ == "__main__":
    a = args(); S.step01_load(synth=a.synth, force=a.force)
