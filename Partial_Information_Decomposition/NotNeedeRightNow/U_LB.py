import numpy
import torch
from typing import Optional
import itertools
from scipy.optimize import minimize
from torch.distributions import Distribution, Normal
from torch.linalg import inv, slogdet
from PID_util import create_cov_matrix, cond_cov



class unique_lb_gauss:
    def __init__(self,T,M1,M2):
        self.T = T - torch.mean(T,dim=0)
        self.M1 = M1 - torch.mean(M1,dim=0)
        self.M2 = M2 - torch.mean(M2,dim=0)
        self.cov_dict = create_cov_matrix(T,M1,M2)

    def _specific_info_quadratic(self, t, ld_ratio, dim, tr_term, B):
        # 0.5[ ln |Σ_M|/|Σ_{M|T}| - d + tr(Σ_M^{-1} Σ_{M|T}) + t^T B t ]
        return 0.5*(ld_ratio - dim + tr_term + t.T @ B @ t)

    def _slogdet(self, A):
        sign, logdet = slogdet(A)
        if sign < 0:
            raise ValueError("Matrix must be positive definite.")
        return logdet
    
    def Smax(self):
         
        # Precompute constants for specific information
        d1 = self.M1.shape[1]
        d2 = self.M2.shape[1]

        conditional__m1 = cond_cov(self.cov_m1,self.cov_t, self.cov_t_m1, self.cov_t_m1.T) #Σ_{M1|T}
        conditional__m2 = cond_cov(self.cov_m2,self.cov_t, self.cov_t_m2, self.cov_t_m2.T) #Σ_{M2|T}

        ratio_1 = self._slogdet(self.cov_m1) - self._slogdet(conditional__m1)
        ratio_2 = self._slogdet(self.cov_m2) - self._slogdet(conditional__m2)

        tr1 = torch.trace(inv(self.cov_m1) @ conditional__m1)
        tr2 = torch.trace(inv(self.cov_m2) @ conditional__m2)

        #Bi​=ΣT−1​ΣTMi​​ΣMi​−1​ΣMi​T​ΣT−1
        B1 = inv(self.cov_t) @ self.cov_t_m1.T @ inv(self.cov_m1) @ self.cov_t_m1 @ inv(self.cov_t)
        B2 = inv(self.cov_t) @ self.cov_t_m2.T @ inv(self.cov_m2) @ self.cov_t_m2 @ inv(self.cov_t)

        #Calculate Imax(T;M1,M2) = Σp(t)max{i}(I(T=t;Mi)) when i is in {1,2}.
        info_T_max = 0
        for t in self.T:
                info_t_m1 = self._specific_info_quadratic(t, ratio_1, d1, tr1, B1)
                info_t_m2 = self._specific_info_quadratic(t, ratio_2, d2, tr2, B2)
                info_t_max = max(info_t_m1, info_t_m2)
                info_T_max += info_t_max
        info_T_max /= self.T.shape[0]

        smax = self.info_t_m1m2 - info_T_max
        return smax

    def unique_lb_gauss(self):
        """This function will compute a lower bound on the true unique information for Gaussian variables
        Unique(T;Mi\Mj) = I(T;Mi|Mj) - Smax(T;M1,M2)
        Where Smax is: I(T;M1,M2) - Imax(T;M1,M2) = I(T;M1,M2) - Σp(t)I(T=t;Mi) when i is in {1,2}.

        input: M1,M2,T are torch tensors of shape (0,d)
        N is the number of observations

        det_ratio_1 = _slogdet(Sigma_M1) - _slogdet(Sig_M1_T)    d is the dimension of each observation.

        output: a lower bound on the unique information Unique(T;Mi\Mj)"""
        


        #Calculate I(T;M1,M2) 0.5*​ln∣ΣT,M1​,M2​​∣∣ΣT​∣∣ΣM1​M2​​∣​
        self.cov_t = self.cov_dict['cov_t']
        self.cov_m1 = self.cov_dict['cov_m1']
        self.cov_m2 = self.cov_dict['cov_m2']
        self.cov_t_m1 = self.cov_dict['cov_t_m1']
        self.cov_t_m2 = self.cov_dict['cov_t_m2']

        self.cov_tm1 = self.cov_dict['cov_tm1']#ΣTM1
        self.cov_tm2 = self.cov_dict['cov_tm2']#ΣTM2

        self.auto_cov_m12 = self.cov_dict['auto_cov_m12']
        cov_matrix = self.cov_dict['full_cov']

        det_cov_t = self._slogdet(self.cov_t)
        det_cross_cov_m1_m2 = self._slogdet(self.auto_cov_m12)
        det_cov_matrix = self._slogdet(cov_matrix)

        self.info_t_m1m2 = 0.5*(det_cov_t + det_cross_cov_m1_m2 - det_cov_matrix)


        #Calculate Smax
        Smax = self.Smax()

        #Calculate I(T;Mi|Mj) = I(T;M1,M2) - I(T;Mj)
        det_cov_m1 = self._slogdet(self.cov_m1)
        det_cov_m2 = self._slogdet(self.cov_m2)
        det_cov_t_m1 = self._slogdet(self.cov_tm1)
        det_cov_t_m2 = self._slogdet(self.cov_tm2)

        info_t_m1 = 0.5*(det_cov_t + det_cov_m1 - det_cov_t_m1)
        info_t_m2 = 0.5*(det_cov_t + det_cov_m2 - det_cov_t_m2)

        info_t_m1_given_m2 = self.info_t_m1m2 - info_t_m2
        info_t_m2_given_m1 = self.info_t_m1m2 - info_t_m1

        unique_lb_m1 = info_t_m1_given_m2 - Smax
        unique_lb_m2 = info_t_m2_given_m1 - Smax

        print(f"\nUnique lower bound for M1: {unique_lb_m1} and unique lower bound for M2: {unique_lb_m2}")

        return unique_lb_m1, unique_lb_m2

