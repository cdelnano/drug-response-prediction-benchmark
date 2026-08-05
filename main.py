import argparse

from models.ridge_regression.model import run as run_ridge
from models.elastic_net.model import run as run_elastic_net
from models.random_forest.model import run as run_random_forest


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run all drug response prediction models."
    )

    parser.add_argument(
        "--treatment-id",
        required=True,
        help="Treatment ID to model (e.g. BRD-K12343256-001-08-9_2.5_HTS)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_ridge(args.treatment_id)
    run_elastic_net(args.treatment_id)
    run_random_forest(args.treatment_id)


if __name__ == "__main__":
    main()