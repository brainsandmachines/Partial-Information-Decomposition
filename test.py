import numpy as np

def rotate_in_random_plane(rng, X, angle_degrees):
    """
    Rotate every row vector of X by angle_degrees in the SAME random 2D plane.
    
    Args:
        rng: np.random.Generator
        X: (n, p) array (rows = samples, cols = features)
        angle_degrees: float
        
    Returns:
        X_rot: (n, p) rotated features
        u, v: the orthonormal plane directions used
    """
    X = np.asarray(X)
    n, p = X.shape

    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)

    # Sample random orthonormal u, v
    u = rng.standard_normal(p)
    u = u / (np.linalg.norm(u) + 1e-12)

    v = rng.standard_normal(p)
    v = v - (v @ u) * u
    v = v / (np.linalg.norm(v) + 1e-12)

    # Coordinates in the plane
    a = X @ u   # (n,)
    b = X @ v   # (n,)

    # Rotate (a,b) -> (a', b')
    a_rot = c * a - s * b
    b_rot = s * a + c * b

    # Put back into R^p: x' = x + (a'-a)u + (b'-b)v
    X_rot = X + np.outer(a_rot - a, u) + np.outer(b_rot - b, v)
    return X_rot, u, v

# Example usage
rng = np.random.default_rng(0)
X = rng.standard_normal((5, 10))
X_rot, u, v = rotate_in_random_plane(rng, X, 30)

print(np.max(np.abs(np.linalg.norm(X, axis=1) - np.linalg.norm(X_rot, axis=1))))
# should be ~1e-10 to 1e-12 (numerical)
