import argparse
import numpy as np
from . import NTS, CNTS

def main():
    parser = argparse.ArgumentParser(description="Run TrustPy trustworthiness evaluation.")
    parser.add_argument("--oracle", type=str, required=True, help="Path to .npy file containing ground truth labels")
    parser.add_argument("--pred", type=str, required=True, help="Path to .npy file containing model predictions (SoftMax)")
    parser.add_argument("--mode", type=str, choices=["nts", "cnts"], required=True, help="Which trust metric to use: nts or cnts")
    parser.add_argument("--trust_spectrum", action="store_true", help="Generate trust spectrum plots")
    parser.add_argument("--no_summary", action="store_true", help="Disable printed summary and CSV export")

    args = parser.parse_args()

    oracle = np.load(args.oracle)
    predictions = np.load(args.pred)

    kwargs = {
        "trust_spectrum": args.trust_spectrum,
        "show_summary": not args.no_summary,
        "export_summary": not args.no_summary
    }

    if args.mode == "nts":
        model = NTS(oracle, predictions, **kwargs)
    elif args.mode == "cnts":
        model = CNTS(oracle, predictions, **kwargs)

    results = model.compute()
    print("\nDone. Results:\n", results)

if __name__ == "__main__":
    main()

