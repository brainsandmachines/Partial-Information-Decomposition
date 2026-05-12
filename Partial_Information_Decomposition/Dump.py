import numpy as np
import torch


def check_supression_effect(vp_results,pid_results):
    """Check for suppression effect in the results.
    Parameters
    ----------
    vp_results : dict
        Results from variance partitioning.
    pid_results : dict
        Results from Partial Information Decomposition.

    Returns
    -------
    None
    """
    R_A = vp_results['R²_X1']
    R_B = vp_results['R²_X2']
    R_AB = vp_results['R²_X12']
    unique_A = vp_results['unique_X1']
    unique_B = vp_results['unique_X2']
    common = vp_results['common']

    unq0 = pid_results['unq0']
    unq1 = pid_results['unq1']
    red = pid_results['red']
    syn = pid_results['syn']

    if np.isclose(R_A,0) or R_A < 0 and unique_A > 0:
        print("\nSuppression effect detected: M1 is a suppressor variable.❗❗❗")
        # if not np.isclose(unq0,0,atol=1e-5):
        #     print("PID fell to supression effect ❌")
        # else: 
        #     if syn > 0:
        #         print("PID did not fall to supression effect and detected synergy ✅✅✅")
        #     else:
        #         print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
            
    elif np.isclose(R_B,0) or R_B < 0 and unique_B > 0:
        print("\nSuppression effect detected: M2 is a suppressor variable.❗❗❗")
        # if not np.isclose(unq1,0,atol=1e-5):
        #     print("PID fell to supression effect ❌")
        # else: 
        #     if syn > 0:
        #         print("PID did not fall to supression effect and detected synergy ✅✅✅")
        #     else:
        #         print("PID did not fall to supression effect ✅ (No synergy detected) ❌")
    else:
        print("\nNo suppression effect detected for VP: One of the unique contributions is zero.")
    return 