import numpy as np
import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import r2_score
from itertools import chain, combinations
from typing import List, Tuple, Union

class CommonalityAnalysis:
    def __init__(self,predictions,target):
        """"args
        target = the vlaues to predict (n,1)
        predictions = The independent values (n,m)
        n: The number of observations
        m: The number of features
        """
        self.target = target
        self.predictions = predictions
        self.models = []
        self.powerset_idx,self.powerset = self.create_powset(self.predictions)
        assert self.target.shape[0] == self.predictions.shape[0], "Incompatible shapes"
        self.all_r2 = 0

    def create_powset(self,X):
        """This function will create the power set for a given predictor
            [1,2,3] --> () (1,) (2,) (3,) ( 1,2) (1,3) (2,3) (1,2,3)"""
        if not isinstance(X,np.ndarray):
            X = np.column_stack(X)
        s = list(X.transpose())
        idxs = range(len(s))
        powerset = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
        powerset_idx = chain.from_iterable(combinations(idxs, r) for r in range(1,len(idxs)+1))

        return list(powerset_idx),list(powerset)

    def r2_for__subset(self,idx_tuple):
        """This function will fit a linear regression model for a given subset of predictors"""
        self.model = LinearRegression()
        Xs = self.predictions[:, idx_tuple] if isinstance(idx_tuple, tuple) else self.predictions[:, (idx_tuple,)]
        model = LinearRegression()  # fit_intercept=True by default
        model.fit(Xs, self.target)
        return model.score(Xs, self.target)



    def r_squared(self):
        self.r2_scores = {}
        for idx_subset in self.powerset_idx:
            r2 = self.r2_for__subset(idx_subset)

            self.r2_scores[idx_subset] = r2
            if len(idx_subset) == len(self.predictions[0]):  # Save the full R^2
                self.all_r2 = r2
        return self

    def _find_intersetion_r2(self):
        """This function will implement the inclusion exclusion formula to get the
        the intersections R^2 between n number of RVs
        i.e. 
        CA{1,2} = R^2(Y;X1) + R^2(Y;X2) - R^2(Y;X1,X2)
        CA{1,2,3} = r2(Y;X1) + R^2(Y;X2) + R^2(Y;X3) - R^2(Y;X1,X2) - R^2(Y;X1,X3) - R^2(Y;X2,X3) + R^2(Y;X1,X2,X3)
        etc..
        NOTE: CA{i} = R^2(Y;Xi)
        """
        if not self.r2_scores:
            self.r_squared()
        self.CA_scores = {}
        for S in self.powerset_idx:
            CA = 0
            subsets = list(chain.from_iterable(combinations(S,r) for r in range(1, len(S)+1)))
            for T in subsets:
                sign = (-1)**(len(subsets) - len(T))
                CA += sign * self.r2_scores[T]
            self.CA_scores[S] = CA
        return self.CA_scores

    def get_unique(self):
        """This function uses _find_intersection_r2 to compute unique contributions of each combination of RVs.
        
        i.e 
        For two random variables:
        U(Y;X1) = R^2(Y;X1) - unique{1,2}

        For three random variables:
        U(Y;X1) = R^2(Y;X1) - unique{1,2} - unique{1,3} - unique{1,2,3}
        
        etc...

        The function takes the real R^2 for n number of RVs, and substracts the CAs
        that the RVs are contained in.

        In general form: 
        Lets R^2(Y;X1,,,Xm) where m < n.
        U(Y;X1,,,Xm) = R^2(Y;X1,,,Xm) - unique{1,,,,m,m+1} - unique{1,,,,m,m+2} - ... - unique{1,,,,,,n}
        """
        self.unique_dict = {}
        for S in list(reversed(self.powerset_idx)):
            self.unique_dict[S] = self.CA_scores[S]
            for key in self.CA_scores.keys():
                if set(S).issubset(key) and S != key:
                    self.unique_dict[S] -= self.unique_dict[key]


        #sanity check: The sum of unique contributions should equal the total R^2
        unique_sum = sum(self.unique_dict.values())

        if not np.isclose(unique_sum, self.all_r2):
            print(f"Warning: Unique values sum to {unique_sum}, not {self.all_r2}")
        else:
            print(f"Unique values sum to {unique_sum}, as expected. ✅")

        return self.unique_dict
    
    def panda_data(self):
        """This function will return a pandas dataframe with the results
        1. Full R^2
        2. Unique contributions
        3. Common contributions
        """
        rows = []
        for k, v in self.CA_scores.items():
            rows.append({"Subset CA scores": k, "R^2": v})
        df_CA = pd.DataFrame(rows)

        rows = []
        for k, v in reversed(self.unique_dict.items()):
            rows.append({"Subset Unique scores": k, "R^2": v})
        df_unique = pd.DataFrame(rows)

        print(df_CA)
        print(df_unique)
        return df_CA, df_unique
    
    

def test_CA():
    rng = np.random.default_rng(0)
    n = 1000000

    S1 = rng.normal(0, 1, n)
    N1 = rng.normal(0, 1, n)
    X1 = S1 + N1
    X2 = N1
    Y  = S1

    X = np.column_stack([X1, X2])
    CA = CommonalityAnalysis(X,Y)
    CA.r_squared()
    CA._find_intersetion_r2()
    CA.get_unique()
    CA.panda_data()

if __name__ == "__main__":
    test_CA()