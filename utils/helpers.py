"""Utility functions for EQ-Negotiator."""

import json
import os
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


def load_scenarios(file_path: str, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """Load scenarios from JSON file."""
    if not os.path.exists(file_path):
        print(f"Scenarios file not found: {file_path}")
        return _default_scenarios()

    with open(file_path, 'r') as f:
        scenarios = json.load(f)

    if n_scenarios:
        scenarios = scenarios[:n_scenarios]

    print(f"Loaded {len(scenarios)} scenarios from {file_path}")
    return scenarios


def _default_scenarios():
    """Default scenarios if no file provided."""
    return [{
        "id": "default_001",
        "product": {"type": "debt_collection", "amount": 15000},
        "seller": {"target_price": 30},
        "buyer": {"target_price": 90},
        "metadata": {
            "outstanding_balance": 15000,
            "creditor_name": "Default Creditor",
            "debtor_name": "Default Debtor",
            "recovery_stage": "Early",
            "cash_flow_situation": "Irregular income",
        }
    }]


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serializer)


def _json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")
