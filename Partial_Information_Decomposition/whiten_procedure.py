import torch

# Force 64-bit precision for large matrix determinants
torch.set_default_dtype(torch.float64)

def whiten_block(Sigma_xx: torch.Tensor,
                 Sigma_xy: torch.Tensor,
                 Sigma_yy: torch.Tensor) -> torch.Tensor:
    """ User's exact PyTorch function """
    Ux = torch.linalg.cholesky(Sigma_xx).T
    Uy = torch.linalg.cholesky(Sigma_yy).T

    tmp = torch.linalg.solve_triangular(Uy.T, Sigma_xy.T, upper=False).T
    K   = torch.linalg.solve_triangular(Ux.T, tmp,        upper=False)

    return K

def run_pqr_simulation():
    difference_list= []
    for i in range(10000):
        torch.manual_seed(i)
        N = 800  # Large sample size to reduce empirical noise
        
        # 1. Define dimensions
        n0, n1, n2 =75, 50, 75  
        
        # 2. Create valid Q and R matrices (Singular values must be < 1)
        buffer = 0.05  # Small buffer to ensure we don't hit the edge of positive definiteness
        Q_raw = torch.randn(n0, n2) * 0.3
        R_raw = torch.randn(n1, n2) * 0.3

        Q_true = Q_raw / (torch.linalg.matrix_norm(Q_raw, ord=2) + buffer)
        R_true = R_raw / (torch.linalg.matrix_norm(R_raw, ord=2) + buffer)
        
        # The derived cross-correlation
        P_true = Q_true @ R_true.T

        # 3. Build the "Whitened" Joint Covariance Matrix (Sigma_7 from your image)
        I_n0 = torch.eye(n0)
        I_n1 = torch.eye(n1)
        I_n2 = torch.eye(n2)
        
        row1 = torch.cat([I_n0, P_true, Q_true], dim=1)
        row2 = torch.cat([P_true.T, I_n1, R_true], dim=1)
        row3 = torch.cat([Q_true.T, R_true.T, I_n2], dim=1)
        
        Sigma_whitened = torch.cat([row1, row2, row3], dim=0)
        
        # 4. Sample the pure, whitened data
        dist = torch.distributions.MultivariateNormal(torch.zeros(n0+n1+n2), Sigma_whitened)
        Z = dist.sample((N,))
        
        Z0 = Z[:, :n0]
        Z1 = Z[:, n0:n0+n1]
        Z2 = Z[:, n0+n1:]

        # 5. "COLOR" THE DATA (Simulate the messy real world)
        # Using absolute values on the diagonal prevents Cholesky crashes for large dims
        L0 = torch.tril(torch.randn(n0, n0)) 
        L0.diagonal().copy_(torch.rand(n0) + 1.0) 
        
        L1 = torch.tril(torch.randn(n1, n1))
        L1.diagonal().copy_(torch.rand(n1) + 1.0)
        
        L2 = torch.tril(torch.randn(n2, n2))
        L2.diagonal().copy_(torch.rand(n2) + 1.0)
        
        # X are our raw, unwhitened observations
        X0 = Z0 @ L0.T
        X1 = Z1 @ L1.T
        X2 = Z2 @ L2.T

        # 6. Calculate Empirical Sample Covariances of the messy data
        X_all = torch.cat([X0, X1, X2], dim=1)
        Sigma_raw = torch.cov(X_all.T, correction=1)  # Sample covariance
        
        # Extract raw blocks
        S00 = Sigma_raw[:n0, :n0]
        S11 = Sigma_raw[n0:n0+n1, n0:n0+n1]
        S22 = Sigma_raw[n0+n1:, n0+n1:]
        
        S01 = Sigma_raw[:n0, n0:n0+n1] # Raw P
        S02 = Sigma_raw[:n0, n0+n1:]   # Raw Q
        S12 = Sigma_raw[n0:n0+n1, n0+n1:] # Raw R

        # 7. APPLY YOUR WHITENING FUNCTION
        P_hat = whiten_block(S00, S01, S11)
        Q_hat = whiten_block(S00, S02, S22)
        R_hat = whiten_block(S11, S12, S22)

        # 8. Check if the recovered relationship holds
        P_recovered_from_QR = Q_hat @ R_hat.T
        difference = torch.norm(P_hat - P_recovered_from_QR).item()
        
        # ---------------------------------------------------------
        # 9. MUTUAL INFORMATION CALCULATION
        # ---------------------------------------------------------
        
        # MI of Raw Data = 0.5 * (log|S00| + log|S11| + log|S22| - log|Sigma_raw|)
        _, logdet_S00 = torch.linalg.slogdet(S00)
        _, logdet_S11 = torch.linalg.slogdet(S11)
        _, logdet_S22 = torch.linalg.slogdet(S22)
        _, logdet_raw = torch.linalg.slogdet(Sigma_raw)
        
        MI_raw = 0.5 * (logdet_S00 + logdet_S11 + logdet_S22 - logdet_raw)
        
        # MI of Whitened Data
        # Reconstruct the empirical whitened joint matrix
        row1_w = torch.cat([I_n0, P_hat, Q_hat], dim=1)
        row2_w = torch.cat([P_hat.T, I_n1, R_hat], dim=1)
        row3_w = torch.cat([Q_hat.T, R_hat.T, I_n2], dim=1)
        Sigma_W_hat = torch.cat([row1_w, row2_w, row3_w], dim=0)
        
        # Because the marginal blocks are Identities, their log determinants are exactly 0.
        # So MI = 0.5 * (0 + 0 + 0 - log|Sigma_W_hat|)
        _, logdet_W_hat = torch.linalg.slogdet(Sigma_W_hat)
        MI_whitened = -0.5 * logdet_W_hat
        
        print("\n--- Simulation Results ---")
        print(f"Norm of difference (P_hat vs Q_hat @ R_hat^T): {difference:.6f}")
        if difference < 0.05:
            print("SUCCESS: Recovered the P = QR^T dependency!\n")
            
        print("--- Mutual Information Preservation ---")
        print(f"MI of Messy Raw Data: {MI_raw.item():.6f}")
        print(f"MI of Whitened Data:  {MI_whitened.item():.6f}")
        
        diff_mi = abs(MI_raw.item() - MI_whitened.item())
        difference_list.append(diff_mi)
        print(f"Difference in MI:     {diff_mi:.10f}")
    print("\n=== Summary of MI Differences Across Simulations ===")
    print(f"Average MI Difference: {torch.mean(torch.tensor(difference_list)):.10f}")
    print(f"Max MI Difference:     {torch.max(torch.tensor(difference_list)):.10f}")
    print(f"Min MI Difference:     {torch.min(torch.tensor(difference_list)):.10f}")

if __name__ == "__main__":
    run_pqr_simulation()