import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from toy_examples.suppression_toy_runner import (
    FEATURE_CORRELATION,
    run_default_factorial_scenarios,
)


# =============================================================================
# 2x3 Factorial Design: SNR (low/high) x Mixing (none/invertible/lossy)
# =============================================================================

def main():
    """Run the 2x3 factorial experiment design."""
    run_default_factorial_scenarios(
        experiment_kind=FEATURE_CORRELATION,
        n=1000,
        p=100,
        seed=42,
    )


if __name__ == '__main__':
    main()
