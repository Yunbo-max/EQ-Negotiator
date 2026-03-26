#!/usr/bin/env python3
"""
EQ-Negotiator: Dynamic Emotional Personas for Credit Negotiation
Main entry point.

Usage:
    python main.py --model_creditor gpt-4o-mini --debtor_persona vanilla --scenarios 5
    python main.py --model_creditor deepseek-7b --debtor_persona angry --iterations 5
    python main.py --debtor_persona all --scenarios 10 --iterations 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_eq_negotiator import main

if __name__ == "__main__":
    main()
