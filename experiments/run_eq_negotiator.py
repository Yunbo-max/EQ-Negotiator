#!/usr/bin/env python3
"""
Run EQ-Negotiator experiments.
Tests across different debtor personas and model configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from models.eq_negotiator import EQNegotiator
from llm.negotiator import EQDebtNegotiator
from utils.helpers import load_scenarios, save_results


def run_experiment(scenarios, args):
    """Run EQ-Negotiator experiment across debtor personas."""

    # Define debtor personas to test
    if args.debtor_persona == "all":
        personas = ["vanilla", "angry", "sad", "fear", "disgust",
                     "threatening", "cheating", "victim", "stonewalling"]
    else:
        personas = [args.debtor_persona]

    all_results = {
        'experiment_type': 'eq_negotiator',
        'model_creditor': args.model_creditor,
        'model_debtor': args.model_debtor,
        'iterations': args.iterations,
        'persona_results': {},
    }

    for persona in personas:
        print(f"\n{'='*60}")
        print(f"DEBTOR PERSONA: {persona.upper()}")
        print(f"{'='*60}")

        # Create shared EQ engine (learns across iterations)
        eq_engine = EQNegotiator()

        persona_negotiations = []

        for iteration in range(args.iterations):
            print(f"\n  Iteration {iteration + 1}/{args.iterations}")

            for i, scenario in enumerate(scenarios):
                print(f"    Scenario {i+1}/{len(scenarios)}: {scenario['id']}")

                negotiator = EQDebtNegotiator(
                    config=scenario,
                    eq_engine=eq_engine,
                    model_creditor=args.model_creditor,
                    model_debtor=args.model_debtor,
                    debtor_persona=persona,
                )

                result = negotiator.run_negotiation(max_dialog_len=args.max_dialog)
                persona_negotiations.append(result)

                outcome = "OK" if result['final_state'] == 'accept' else "FAIL"
                days = result.get('collection_days', 'N/A')
                rounds = result['negotiation_rounds']
                print(f"      {outcome} | Days: {days} | Rounds: {rounds}")

        # Compute persona summary
        successful = [r for r in persona_negotiations if r['final_state'] == 'accept']
        success_rate = len(successful) / len(persona_negotiations) if persona_negotiations else 0

        if successful:
            avg_days = np.mean([r['collection_days'] for r in successful if r['collection_days']])
            avg_rounds = np.mean([r['negotiation_rounds'] for r in successful])
            avg_multiple = np.mean([
                r['collection_days'] / max(1, r['creditor_target_days'])
                for r in successful if r['collection_days']
            ])
        else:
            avg_days = avg_rounds = avg_multiple = 0

        summary = {
            'success_rate': success_rate,
            'avg_collection_days': float(avg_days),
            'avg_negotiation_rounds': float(avg_rounds),
            'avg_debt_multiple': float(avg_multiple),
            'total_negotiations': len(persona_negotiations),
            'successful': len(successful),
        }

        all_results['persona_results'][persona] = {
            'summary': summary,
            'negotiations': persona_negotiations,
        }

        print(f"\n  {persona.upper()} Summary:")
        print(f"    Success Rate: {success_rate:.0%}")
        print(f"    Avg Collection Days: {avg_days:.1f}")
        print(f"    Avg Debt Multiple: {avg_multiple:.2f}x")
        print(f"    Avg Rounds: {avg_rounds:.1f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    result_file = f"{out_dir}/eq_negotiator_{timestamp}.json"
    save_results(all_results, result_file)
    print(f"\nResults saved to: {result_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run EQ-Negotiator Experiments")
    parser.add_argument("--scenarios", type=int, default=3, help="Number of scenarios")
    parser.add_argument("--iterations", type=int, default=3, help="Learning iterations")
    parser.add_argument("--max_dialog", type=int, default=30, help="Max dialog turns")
    parser.add_argument("--model_creditor", default="gpt-4o-mini", help="Creditor model")
    parser.add_argument("--model_debtor", default="gpt-4o-mini", help="Debtor model")
    parser.add_argument("--debtor_persona", default="vanilla",
                        choices=["vanilla", "angry", "sad", "fear", "disgust",
                                 "threatening", "cheating", "victim", "stonewalling", "all"],
                        help="Debtor persona strategy")
    parser.add_argument("--out_dir", default="results", help="Output directory")

    args = parser.parse_args()

    scenarios = load_scenarios("config/scenarios.json", n_scenarios=args.scenarios)
    results = run_experiment(scenarios, args)

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for persona, data in results['persona_results'].items():
        s = data['summary']
        print(f"  {persona:15s} | SR: {s['success_rate']:.0%} | "
              f"Multiple: {s['avg_debt_multiple']:.2f}x | "
              f"Rounds: {s['avg_negotiation_rounds']:.1f}")


if __name__ == "__main__":
    main()
