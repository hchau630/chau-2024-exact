import logging
from collections.abc import Hashable, Iterable

import torch
import pandas as pd
import tdfl

from niarb.tensors import categorical

logger = logging.getLogger(__name__)

__all__ = ["TensorDataFrameAnalysis", "PandasDataFrameAnalysis"]


class DataFrameAnalysis(torch.nn.Module):
    def __init__(self, x: pd.DataFrame, y: Hashable):
        if not isinstance(x, pd.DataFrame):
            raise TypeError(f"x must be a pd.DataFrame, but got {type(x)=}.")

        if not isinstance(y, Hashable):
            raise TypeError(f"y must be hashable, but got {type(y)=}.")

        if not len(x.drop_duplicates()) == len(x):
            raise ValueError("x must consist of unique rows.")

        super().__init__()
        self.x = x.copy()
        self.y = y


class TensorDataFrameAnalysis(DataFrameAnalysis):
    def __init__(
        self,
        x: pd.DataFrame,
        y: str,
        estimator: str = "mean",
        sem: str | Iterable[str] | None = None,
    ):
        r"""Perform groupby aggregations on a tdfl.DataFrame.

        Args:
            x: Labels with which data is grouped on.
            y: Column on which statistics are calculated.
            estimator (optional): {"mean", "median"}. Estimator to use for calculating
              statistics. If "median" and `sem` is not None, the standard error of the
              median is calculated by multiplying the standard error of the mean by
              $\sqrt{\pi/2}$ (normality assumption).
            sem (optional): If not None, computes SEM across specified columns.

        """
        super().__init__(x, y)

        intervals = {}
        for k, v in x.items():
            if v.dtype.name == "category" and isinstance(
                v.cat.categories, pd.IntervalIndex
            ):
                intervals[k] = v.cat.categories
            if v.dtype.name == "interval":
                intervals[k] = pd.IntervalIndex(v)
        self.intervals = intervals

        if estimator not in {"mean", "median"}:
            raise ValueError(f"Unknown estimator: {estimator=}.")
        self.estimator = estimator

        if isinstance(sem, str):
            sem = [sem]
        if isinstance(sem, Iterable):
            sem = list(sem)
        if sem is not None and any(k in x.columns for k in sem):
            raise ValueError(f"{sem=} must not be in {list(x.columns)=}.")
        self.sem = sem

    def forward(self, df: tdfl.DataFrame) -> tdfl.DataFrame:
        df = df.copy()

        # Cut the interval columns for grouping
        for k, v in self.intervals.items():
            # Note: need to convert to float due to lib.pt._bin_numbers bug with Long input.
            # labels=False with missing_rep=-1 yields the codes of the categories.
            df[k] = tdfl.cut(df[k].float(), v, labels=False, missing_rep=-1)
            df[k] = categorical.as_tensor(df[k], categories=list(v) + [torch.nan])

        # Filter out all invalid groups first, which would make groupby faster
        df = df[:, (torch.stack([df[k] for k in self.x.columns]) != -1).all(dim=0)]

        # Perform groupby operations
        if self.sem is not None:
            df = df.groupby(list(self.x.columns) + self.sem)[self.y].agg(self.estimator)
            out = df.groupby(list(self.x.columns)).agg(
                **{self.y: (self.y, self.estimator), f"{self.y}_se": (self.y, "sem")}
            )
            if self.estimator == "median":
                out[f"{self.y}_se"] = out[f"{self.y}_se"] * (torch.pi / 2) ** 0.5
        else:
            # Note: df.groupby() is likely a bottleneck due to the slow torch.unqiue call
            out = df.groupby(list(self.x.columns))[self.y].agg(self.estimator)

        # Convert CategoricalTensors to numpy arrays before merging
        for k, v in out.items():
            if isinstance(v, categorical.CategoricalTensor):
                out[k] = v.detach().cpu().numpy()

        out = out.merge(self.x, how="right")

        if out[self.y].isnan().any():
            logger.warning(
                f"NaNs detected in analysis output\n:{out[:, out[self.y].isnan()]}"
            )

        return out


class PandasDataFrameAnalysis(DataFrameAnalysis):
    def __init__(self, x, y):
        noninterval_category_columns = set()
        interval_columns = set()

        for k, v in x.items():
            if v.dtype.name == "category":
                if isinstance(v.cat.categories, pd.IntervalIndex):
                    interval_columns.add(k)
                else:
                    noninterval_category_columns.add(k)

        super().__init__(x, y)

        self.noninterval_category_columns = noninterval_category_columns
        self.interval_columns = interval_columns

    def forward(self, df):
        for k in self.interval_columns:
            df[k] = pd.cut(df[k], self.x[k].cat.categories)
        df = df.groupby(list(self.x.columns), as_index=False)[self.y].mean()
        for k in self.noninterval_category_columns:
            df[k] = pd.Categorical.from_codes(
                df[k], categories=self.x[k].cat.categories
            )

        return self.x.merge(df, how="left")
