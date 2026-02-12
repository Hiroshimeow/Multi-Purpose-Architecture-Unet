
import os
import torch
import argparse
import sys

# --- How to Run ---
# python scripts/calculate_model_parameters.py -c 4 9 -c 5 6
# The command above calculates params for (depth=4, initial_filters=9) and (depth=5, initial_filters=6)

def main():
    """
    Calculates the parameter count for one or more UNetBase model configurations.
    """
    parser = argparse.ArgumentParser(
        description="Calculate the parameter count for various UNetBase model configurations.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--config', 
        nargs=2, 
        type=int, 
        action='append',
        metavar=('DEPTH', 'INITIAL_FILTERS'),
        help="Specify a (depth, initial_filters) combination to calculate. Can be used multiple times.\n" \
             "Example: -c 4 9 -c 5 6"
    )
    
    args = parser.parse_args()

    if not args.config:
        parser.print_help()
        print("\nError: No configurations provided. Please use the -c flag.", file=sys.stderr)
        sys.exit(1)

    # This assumes the script is run from the project root,
    # allowing the 'src' module to be found.
    try:
        from src.models.unet_base import UNetBase
    except ImportError:
        print("Error: Could not import UNetBase from 'src.models.unet_base'.", file=sys.stderr)
        print("Please ensure you are running this script from the project's root directory.", file=sys.stderr)
        sys.exit(1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n--- Calculating Parameter Counts for UNetBase ---")
    print("-" * 45)

    # Common parameters for instantiation that don't affect the result
    in_channels = 3 
    num_classes = 5

    for depth, initial_filters in args.config:
        try:
            model = UNetBase(
                in_channels=in_channels, 
                num_classes=num_classes, 
                initial_filters=initial_filters, 
                depth=depth
            )
            params_m = count_parameters(model) / 1_000_000
            print(f"d={depth}, i={initial_filters}  =>  Params: {params_m:.2f}M")
        except Exception as e:
            print(f"d={depth}, i={initial_filters}  =>  Error: {e}", file=sys.stderr)

    print("-" * 45)

if __name__ == "__main__":
    main()
