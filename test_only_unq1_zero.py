#!/usr/bin/env python
"""
Test script for the only_unq1_zero simulation mode
"""

import torch
import sys
from pathlib import Path

# Setup paths
root = Path(__file__).resolve().parent
sys.path.append(str(root / "Partial_Information_Decomposition"))
sys.path.append(str(root / "Partial_Information_Decomposition/Idep_Simulations"))

from Idep_Simulations.wrapper_M7_M8_models import make_random_true_cov, create_m7_cov


def test_only_unq1_zero_mode():
    """Test the only_unq1_zero mode"""
    
    print("=" * 70)
    print("Testing only_unq1_zero simulation mode")
    print("=" * 70)
    
    # Configuration for the test
    config = {
        'device': 'cpu',
        'n0': 3,   # Dimension of X0
        'n1': 2,   # Dimension of X1
        'n2': 2,   # Dimension of Y
        'q_scale': 1.0,
        'r_scale': 1.0,
        'mode': 'only_unq1_zero',
        'max_tries': 50,  # Smaller number for quick test
        'de_seed': 42,
        'de_maxiter': 300,  # Smaller for quick test
        'de_tolerance': 1e-8,
        'unq1_threshold': 1e-4,
        'red_syn_margin': 1e-5,
    }
    
    print("\nConfig:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nAttempting to create only_unq1_zero covariance...")
    
    try:
        m8_cov, m7_cov = make_random_true_cov(config)
        
        print("\n✓ SUCCESS: Created only_unq1_zero covariance matrices!")
        print(f"\nM8 covariance shape: {m8_cov.shape}")
        print(f"M7 covariance shape: {m7_cov.shape}")
        
        # Check positive definiteness
        eigvals_m8 = torch.linalg.eigvalsh(m8_cov)
        eigvals_m7 = torch.linalg.eigvalsh(m7_cov)
        
        print(f"\nM8 min eigenvalue: {torch.min(eigvals_m8):.6e}")
        print(f"M7 min eigenvalue: {torch.min(eigvals_m7):.6e}")
        
        if torch.min(eigvals_m8) > 1e-10 and torch.min(eigvals_m7) > 1e-10:
            print("✓ Both covariances are positive definite")
        else:
            print("✗ WARNING: Covariances may not be sufficiently positive definite")
        
        return True
        
    except RuntimeError as e:
        print(f"\n✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_only_unq1_zero_mode()
    
    print("\n" + "=" * 70)
    if success:
        print("Test PASSED ✓")
    else:
        print("Test FAILED ✗")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
