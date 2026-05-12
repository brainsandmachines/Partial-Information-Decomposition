import numpy as np

def theoretical_covariance(p: int, noise_std: float, order=("X1", "X2", "Y")):
    """
    Theoretical covariance for:

        Y  = R + U + eps_y
        X1 = R + U + N
        X2 = R     + N

    with R, U ~ N(0, I), N, eps_y ~ N(0, noise_std^2 I),
    mutually independent.
    """

    s2 = noise_std ** 2
    I = np.eye(p)

    cov_blocks = {
        ("X1", "X1"): (2 + s2) * I,
        ("X2", "X2"): (1 + s2) * I,
        ("Y",  "Y"):  (2 + s2) * I,

        ("X1", "X2"): (1 + s2) * I,
        ("X1", "Y"):  2 * I,
        ("X2", "Y"):  1 * I,
    }

    for (a, b), block in list(cov_blocks.items()):
        cov_blocks[(b, a)] = block

    Sigma = np.block([
        [cov_blocks[(a, b)] for b in order]
        for a in order
    ])

    return Sigma


def simulate_process(n: int, p: int, noise_std: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    R = rng.standard_normal((n, p))
    U = rng.standard_normal((n, p))
    N = noise_std * rng.standard_normal((n, p))
    eps_y = noise_std * rng.standard_normal((n, p))

    Y = R + U + eps_y
    X1 = R + U + N
    X2 = R + N

    return X1, X2, Y


def validate_covariance(n=1_000_000, p=5, noise_std=0.3, seed=0):
    rng = np.random.default_rng(seed)

    X1, X2, Y = simulate_process(n, p, noise_std, rng)

    # Concatenate in the same order as theoretical_covariance
    Z = np.concatenate([X1, X2, Y], axis=1)

    empirical = np.cov(Z, rowvar=False, bias=False)
    theoretical = theoretical_covariance(p, noise_std)

    abs_err = np.abs(empirical - theoretical)

    print("Empirical covariance:")
    print(empirical)

    print("\nTheoretical covariance:")
    print(theoretical)

    print("\nMax absolute error:", abs_err.max())
    print("Mean absolute error:", abs_err.mean())

    return empirical, theoretical, abs_err


empirical, theoretical, abs_err = validate_covariance(
    n=1_000_000,
    p=3,
    noise_std=0.3,
    seed=123,
)


def validate_per_feature_covariance(n=1_000_000, p=5, noise_std=0.3, seed=0):
    rng = np.random.default_rng(seed)

    X1, X2, Y = simulate_process(n, p, noise_std, rng)

    target = np.array([
        [2 + noise_std**2, 1 + noise_std**2, 2],
        [1 + noise_std**2, 1 + noise_std**2, 1],
        [2,                1,                2 + noise_std**2],
    ])

    errors = []

    for j in range(p):
        Zj = np.column_stack([X1[:, j], X2[:, j], Y[:, j]])
        empirical_j = np.cov(Zj, rowvar=False, bias=False)
        err_j = empirical_j - target
        errors.append(err_j)

        print(f"\nFeature {j}")
        print("Empirical:")
        print(empirical_j)
        print("Error:")
        print(err_j)

    errors = np.array(errors)

    print("\nTarget covariance:")
    print(target)

    print("\nMax absolute error across features:", np.abs(errors).max())
    print("Mean absolute error across features:", np.abs(errors).mean())

    return errors


errors = validate_per_feature_covariance(
    n=1_000_000,
    p=5,
    noise_std=0.3,
    seed=123,
)


