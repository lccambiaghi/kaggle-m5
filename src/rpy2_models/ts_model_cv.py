from abc import abstractmethod
from typing import Union

from rpy2.rinterface import NULLType
from rpy2 import robjects
from rpy2.robjects import NULL
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from rpy2_models import r_models_functions
from rpy2_models.ts_model import TimeSeriesModel, ARIMAX, ExponentialSmoothing
from rpy2_models.exceptions import ModelStructureNotFoundError, CrossValidationScoreError

VAL_WEEKS = 26
MAX_HORIZON = 8


def rmse(errors: np.ndarray) -> float:
    return np.sqrt(np.nanmean(errors ** 2))


class TimeSeriesModelCV(TimeSeriesModel):
    @abstractmethod
    def find_model_structure(self, y_train: np.ndarray, xreg_train: Union[np.ndarray, NULLType]):
        """
        This method identifies the best model structure given the training data (example: auto_arima).
        The fitted structure is to be evaluated on the validation data in a time series cross validation fashion.

        :param y_train: response time series
        :param xreg_train: external regressors
        :return: fitted model object
        """
        pass

    def cv_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        This method computes the cross_validation_score for this TimeSeriesModel evaluated on X and y.
        First the model structure is identified by calling self.find_model_structure(y_train, X_train).
        Parameters are estimated and a forecast is produced for each week in the validation set, for each forecast horizon.
        At the moment, what is returned is the four_weeks_ahead_rmse.

        :param X: external regressors
        :param y: response time series
        :return:
        """
        # TODO: n_train = len(y) - VAL_WEEKS - MAX_HORIZON would make sure we have at least VAL_WEEKS predictions for far horizons
        n_train = len(y) - VAL_WEEKS - 1
        y_train = y[:n_train, :]
        # NULL is the R type for None
        X_train = X[:n_train, :] if X.size else NULL

        try:
            self.model = self.find_model_structure(y_train, X_train)
        except Exception:
            raise ModelStructureNotFoundError
        try:
            self.cv_errors = self.compute_cv_errors(y, X)
        except Exception:
            raise CrossValidationScoreError

        four_weeks_ahead_rmse = rmse(self.cv_errors[:, 3])

        return four_weeks_ahead_rmse

    def compute_cv_errors(self, y: np.ndarray, X: Union[np.ndarray, NULLType]):
        cv_errors = np.ndarray(shape=(VAL_WEEKS, MAX_HORIZON))

        cv = RollingTSSplit(max_horizon=MAX_HORIZON, val_size=VAL_WEEKS)
        folds = list(cv.split(y))

        for t, (train, test) in enumerate(folds):
            y_train, y_test = y[train], y[test].flatten()
            X_train = X[train] if X.size else robjects.NULL
            X_test = X[test] if X.size else robjects.NULL

            self.model = self.fit(y_train, X_train)
            df_forecast = self.predict(MAX_HORIZON, X_test)
            cv_errors[t, :] = df_forecast['prediction'].values - y_test

        return cv_errors


class ARIMAXCV(TimeSeriesModelCV, ARIMAX):
    def find_model_structure(self, y_train: np.ndarray, xreg_train: Union[np.ndarray, NULLType]):
        self.model = r_models_functions.auto_arima(y_train, xreg_train, self.seasonal)
        self.is_fitted_ = True
        return self.model


class ExponentialSmoothingCV(TimeSeriesModelCV, ExponentialSmoothing):
    def find_model_structure(self, y_train: np.ndarray, xreg_train: NULLType):
        self.model = r_models_functions.auto_ets(y_train)
        self.is_fitted_ = True
        return self.model


# TODO write some tests, make it more robust
class RollingTSSplit(TimeSeriesSplit):
    # TODO turn into method?
    def __init__(self, max_horizon, val_size):
        self.val_size = val_size
        self.max_horizon = max_horizon

    def split(self, y):
        n_samples = y.shape[0]
        n_val_samples = self.val_size
        n_train_samples = n_samples - n_val_samples - self.max_horizon - 1

        test_size = self.max_horizon
        # TODO make this more clear
        test_starts = range(n_train_samples + 1,
                            n_samples - self.max_horizon, 1)

        indices = np.arange(n_samples)
        for test_start in test_starts:
                yield (indices[:test_start],
                       indices[test_start:test_start+test_size])
