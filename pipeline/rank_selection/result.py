"""One output format for every rank-selection method."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RankResult:
    """A method's rank curve and its selected rank.

    Fields:
        method: str, method label including its options, e.g.
            ``"eigenvector[basis=pca,folds=5]"``.
        dataset: str, dataset label.
        seed: int or None, the method's fold/permutation seed.
        selected_rank: int, the rank the method selects by its own rule.
        selection_rule: str, how selected_rank was chosen.
        score_name: str, what ``score`` measures.
        rank: np.ndarray (n_candidates,), candidate ranks.
        score: np.ndarray (n_candidates,), the criterion at each rank.
        standard_error: np.ndarray (n_candidates,) or None, across folds.
        fold_scores: np.ndarray (n_folds, n_candidates) or None.
        best_rank: int or None, the rank optimizing ``score``.
        one_se_rank: int or None, the smallest rank within one standard error
            of the optimum.
        true_rank: int or None, toy ground truth.
        extra_curves: dict[str, np.ndarray], other per-rank columns, each
            shaped (n_candidates,).
        info: dict[str, Any], scalar metadata.
    """

    method: str
    dataset: str
    seed: int | None
    selected_rank: int
    selection_rule: str
    score_name: str
    rank: np.ndarray
    score: np.ndarray
    standard_error: np.ndarray | None = None
    fold_scores: np.ndarray | None = None
    best_rank: int | None = None
    one_se_rank: int | None = None
    true_rank: int | None = None
    extra_curves: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_table(self) -> pd.DataFrame:
        """Return one row per candidate rank.

        Input: none.
        Output: pd.DataFrame with ``rank``, ``score``, optional
            ``standard_error``, ``fold_{i}_score`` columns, the extra curves,
            and Boolean ``is_selected``, ``is_best``, ``is_one_se`` columns.
        """
        columns: dict[str, Any] = {"rank": self.rank, "score": self.score}
        if self.standard_error is not None:
            columns["standard_error"] = self.standard_error
        if self.fold_scores is not None:
            # (n_folds, n_candidates) -> n_folds columns of shape (n_candidates,)
            for fold_index, fold in enumerate(self.fold_scores, start=1):
                columns[f"fold_{fold_index}_score"] = fold
        for name, curve in self.extra_curves.items():
            if np.shape(curve) != np.shape(self.rank):
                raise ValueError(f"Extra curve {name!r} must have shape {np.shape(self.rank)}.")
            columns[name] = curve
        columns["is_selected"] = self.rank == self.selected_rank
        columns["is_best"] = self.rank == self.best_rank
        columns["is_one_se"] = self.rank == self.one_se_rank
        return pd.DataFrame(columns)

    def summary(self) -> dict[str, Any]:
        """Return the scalar fields as one summary row.

        Input: none.
        Output: dict[str, Any] with method, dataset, seed, ranks, score name,
            selection rule, the true rank, and the ``info`` entries.
        """
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "selected_rank": self.selected_rank,
            "best_rank": self.best_rank,
            "one_se_rank": self.one_se_rank,
            "true_rank": self.true_rank,
            "score_name": self.score_name,
            "selection_rule": self.selection_rule,
            **self.info,
        }


def one_standard_error_rank(
    rank: np.ndarray,
    mean_score: np.ndarray,
    standard_error: np.ndarray,
    minimize: bool = True,
) -> int:
    """Return the smallest rank whose score is within one SE of the optimum.

    Inputs:
        rank: np.ndarray (n_candidates,), ascending candidate ranks.
        mean_score: np.ndarray (n_candidates,), may contain NaN (ignored).
        standard_error: np.ndarray (n_candidates,).
        minimize: bool, True when lower scores are better.
    Output: int, the selected rank.
    """
    sign = 1.0 if minimize else -1.0
    signed = sign * np.asarray(mean_score, dtype=float)  # (n_candidates,)
    best = int(np.nanargmin(signed))
    threshold = signed[best] + standard_error[best]
    within = np.flatnonzero(signed <= threshold)  # NaN compares False
    return int(rank[within[0]])
