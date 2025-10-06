from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import typing as tp

from jaxtyping import (
    Float,
)
import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.indexes.datetimes import DatetimeIndex

from causal_validation.types import InterventionTypes


@dataclass
class Dataset:
    """A causal inference dataset containing outputs that are the
    metric of interest, covariates, and treatment status.

    Attributes:
        Y (Float[np.ndarray, "T N"]): Matrix of outputs that are
            the metric of interest for T time points and total
            N units.
        D (Float[np.ndarray, "T N"]): Matrix of treatment status
            for all N units at every T time points. D can be a
            matrix of binary or continuous values depending on
            treatment type.
        X (Optional[Float[np.ndarray, "T N D"]]): A tensor of
            D-dimensional covariates for T time periods and
            N units.
        _start_date: Start date for time indexing
        _name: Optional name identifier for the dataset
    """

    Y: Float[np.ndarray, "T N"]
    D: Float[np.ndarray, "T N"]
    X: tp.Optional[Float[np.ndarray, "T N D"]] = None
    _start_date: dt.date = dt.date(year=2023, month=1, day=1)
    # counterfactual: tp.Optional[Float[np.ndarray, "M 1"]] = None
    # synthetic: tp.Optional[Float[np.ndarray, "M 1"]] = None
    _name: tp.Optional[str] = None

    def to_df(
        self, index_start: dt.date = dt.date(year=2023, month=1, day=1)
    ) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame with multi-level columns.

        Args:
            index_start (date): Start date for the DataFrame index.

        Returns:
            DataFrame: A pandas DataFrame with time index and multi-level
                columns for units and variables.
        """
        index = self._get_index(index_start)
        unit_cols = self._get_columns()

        var_cols = ["Y", "D"]
        if self.X is not None:
            var_cols.extend([f"X{i}" for i in range(self.n_covariates)])

        col_tuples = [(unit, var) for unit in unit_cols for var in var_cols]
        multi_cols = pd.MultiIndex.from_tuples(col_tuples)

        unit_data = []
        for i in range(self.n_units):
            unit_data.append(self.Y[:, i : i + 1])
            unit_data.append(self.D[:, i : i + 1])
            if self.X is not None:
                unit_data.append(self.X[:, i, :])

        combined_data = np.concatenate(unit_data, axis=1)
        return pd.DataFrame(combined_data, index=index, columns=multi_cols)

    @property
    def n_post_intervention(self) -> list[int]:
        """Number of post-intervention time points for each unit.

        Returns:
            list[int]: List of post-intervention counts for each unit.
        """
        binary_assignments = -1 * ((self.D == 0).astype(int) - 1)
        binary_assignment_counts = (binary_assignments.sum(axis=0)).tolist()
        return binary_assignment_counts

    @property
    def n_pre_intervention(self) -> list[int]:
        """Number of pre-intervention time points for each unit.

        Returns:
            list[int]: List of pre-intervention counts for each unit.
        """
        is_zero = (self.D == 0).astype(int)
        pre_intervention_counts = (is_zero.sum(axis=0)).tolist()
        return pre_intervention_counts

    @property
    def n_units(self) -> int:
        """Total number of units in the dataset.

        Returns:
            int: Number of units.
        """
        return self.Y.shape[1]

    @property
    def n_control_units(self) -> int:
        """Number of control units in the dataset.

        Returns:
            int: Number of control units (units with treatment
                assignment 0 in all time points).
        """
        return len(self.control_unit_indices)

    @property
    def n_treated_units(self) -> int:
        """Number of treated units in the dataset.

        Returns:
            int: Number of treated units (units with non-zero
                treatment assignment at least once.
        """
        return len(self.treated_unit_indices)

    @property
    def n_timepoints(self) -> int:
        """Total number of time points in the dataset.

        Returns:
            int: Number of time points.
        """
        return self.Y.shape[0]

    @property
    def n_covariates(self) -> int:
        """Covariate dimensionality.

        Returns:
            int: Number of covariates, or 0 if no covariates exist.
        """
        if self.X is not None:
            return self.X.shape[2]
        else:
            return 0

    @property
    def control_unit_indices(self) -> list[int]:
        """Indices of control units (units that never receive treatment).

        Returns:
            list[int]: List of control unit indices.
        """
        return [i for i in range(self.n_units) if all(self.D[:, i] == 0)]

    @property
    def treated_unit_indices(self) -> list[int]:
        """Indices of treated units (units that receive treatment at some point).

        Returns:
            list[int]: List of treated unit indices.
        """
        return [i for i in range(self.n_units) if any(self.D[:, i] != 0)]

    @property
    def control_unit_outputs(
        self,
    ) -> Float[np.ndarray, "{self.n_timepoints} {self.n_control_units}"]:
        """Output values for control units (units that never receive treatment)
            only.

        Returns:
            Float[np.ndarray, "n_timepoints n_control_units"]:
                Array of outputs for control units across all time points.
        """
        return self.Y[:, self.control_unit_indices]

    @property
    def treated_unit_outputs(
        self,
    ) -> Float[np.ndarray, "{self.n_timepoints} {self.n_treated_units}"]:
        """Output values for treated (units that receive treatment at some point)
            units only.

        Returns:
            Float[np.ndarray, "n_timepoints n_control_units"]:
                Array of outputs for treated units across all time points.
        """
        return self.Y[:, self.treated_unit_indices]

    @property
    def control_unit_covariates(
        self,
    ) -> tp.Optional[
        Float[
            np.ndarray, "{self.n_timepoints} {self.n_control_units} {self.n_covariates}"
        ]
    ]:
        """Covariate values for control units (units that never receive treatment)
            only.

        Returns:
            Float[np.ndarray, "n_timepoints n_control_units n_covariates"]:
                Array of covariates for control units, or None if no covariates
                exist.
        """
        if self.X is not None:
            return self.X[:, self.control_unit_indices, :]

    @property
    def treated_unit_covariates(
        self,
    ) -> tp.Optional[
        Float[
            np.ndarray, "{self.n_timepoints} {self.n_treated_units} {self.n_covariates}"
        ]
    ]:
        """Covariate values for treated units (units that receive treatment at some
            point) only.

        Returns:
            Float[np.ndarray, "n_timepoints n_control_units n_covariates"]:
                Array of covariates for treated units, or None if no covariates
                exist.
        """
        if self.X is not None:
            return self.X[:, self.treated_unit_indices, :]

    @property
    def full_index(self) -> DatetimeIndex:
        return self._get_index(self._start_date)

    def treatment_date(self, unit_idx: int) -> tp.Optional[Timestamp]:
        """Treatment start date for a specific unit.

        Args:
            unit_idx: Index of the unit.

        Returns:
            Optional[Timestamp]: When treatment begins for the
                specified unit.
        """
        idxs = self.full_index
        treatment_idx = self.n_pre_intervention[unit_idx]
        if treatment_idx >= len(idxs):
            return None
        else:
            return idxs[self.n_pre_intervention[unit_idx]]

    def get_index(self, period: InterventionTypes, unit_idx: int) -> DatetimeIndex:
        if period == "pre-intervention":
            return self.full_index[: self.n_pre_intervention[unit_idx]]
        elif period == "post-intervention":
            return self.full_index[self.n_pre_intervention[unit_idx] :]
        else:
            return self.full_index

    def _get_columns(self) -> tp.List[str]:
        colnames = [f"U{i}" for i in range(self.n_units)]
        return colnames

    def _get_index(self, start_date: dt.date) -> DatetimeIndex:
        return pd.date_range(start=start_date, freq="D", periods=self.n_timepoints)

    def __eq__(self, other: Dataset) -> bool:
        if self.Y.shape == other.Y.shape:
            Y_check = np.allclose(self.Y, other.Y)
        else:
            Y_check = False
        if self.D.shape == other.D.shape:
            D_check = np.allclose(self.D, other.D)
        else:
            D_check = False
        if self.X is not None:
            if other.X is not None:
                if self.X.shape == other.X.shape:
                    X_check = np.allclose(self.X, other.X)
                else:
                    X_check = False
            else:
                X_check = False
        elif other.X is None:
            X_check = True
        else:
            X_check = False

        return all([Y_check, D_check, X_check])

    @property
    def _slots(self) -> tp.Dict[str, int]:
        return {
            "n_units": self.n_units,
            "n_timepoints": self.n_timepoints,
            "n_covariates": self.n_covariates,
        }

    @property
    def name(self) -> tp.Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value


@dataclass
class DatasetContainer:
    """Container for multiple Dataset instances.

    Attributes:
        datasets (List[Dataset]): List of Dataset instances.
        names (Optional[tp.List[str]]): List of names for the datasets.
    """

    datasets: tp.List[Dataset]
    names: tp.Optional[tp.List[str]] = None

    def __post_init__(self):
        self.n_datasets = len(self.datasets)
        if self.names is None:
            names = []
            for idx, dataset in enumerate(self.datasets):
                if dataset.name:
                    names.append(dataset.name)
                else:
                    names.append(f"Dataset {idx}")
            self.names = names

    def __iter__(self) -> tp.Iterator[Dataset]:
        return iter(self.datasets)

    def as_dict(self) -> tp.Dict[str, Dataset]:
        dict_result = {}
        for n, d in zip(self.names, self.datasets, strict=True):
            dict_result[n] = d
        return dict_result

    def __len__(self) -> int:
        return len(self.datasets)
