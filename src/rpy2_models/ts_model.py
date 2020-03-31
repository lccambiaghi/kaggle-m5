from abc import abstractmethod
from pandas import DataFrame
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from rpy2.robjects import numpy2ri, pandas2ri
import logging
from typing import Union
from rpy2.rinterface import NULLType
from rpy2.robjects import NULL
from sklearn.utils.validation import check_is_fitted

from rpy2_models import r_models_functions
from rpy2_models.exceptions import ModelNotFitError

numpy2ri.activate()
pandas2ri.activate()

class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """
    TimeSeriesModel using find_model_structure, fit and predict methods implemented in R.

    During model selection the pipeline is find_model_structure -> cross_val_score
    When forecasting the pipeline is fit -> predict
    """

    def __init__(self):
        self.model = None
        self.is_fitted_ = False
        self.cv_errors: np.ndarray
        self.fitted: np.ndarray
        self.coef_: DataFrame
        self.sigma2: float

    @abstractmethod
    def fit(self, y_train: np.ndarray, xreg_train: Union[np.ndarray, NULLType]):
        """
        This method is called to estimate the model parameters, given the training data.
        If the model has already been fit before, coefficients get reestimated.
        This method is called when forecasting, where the selected models get reestimated on the full training data.

        :param y_train: response time series
        :param xreg_train: external regressors
        :return: fitted model object
        """
        pass

    @abstractmethod
    def predict(self, horizon_max: int, xreg_test: Union[np.ndarray, NULLType]) -> DataFrame:
        pass

    def fit_predict(
        self,
        y_train: np.ndarray,
        test_index: DataFrame,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            self.model = self.fit(y_train, xreg_train)
        except Exception:
            raise ModelNotFitError
        df_forecast = self.predict(test_index.shape[0], xreg_test)
        df_forecast["prediction_date"] = test_index
        return df_forecast


class ARIMAX(TimeSeriesModel):
    def __init__(
        self, ar_order=0, d_order=0, ma_order=0, seasonal=False
    ):
        super().__init__()
        self.ar_order = ar_order
        self.d_order = d_order
        self.ma_order = ma_order
        self.seasonal = seasonal

    def fit(self, y_train: np.ndarray, xreg_train: Union[np.ndarray, NULLType]) -> TimeSeriesModel:
        if self.is_fitted_:
            self.model = r_models_functions.refit_arima(self.model, y_train, xreg_train)
        else:
            order_vector = np.array((self.ar_order, self.d_order, self.ma_order))
            self.model = r_models_functions.fit_arima(order_vector, y_train, xreg_train)
            self.is_fitted_ = True

        self.coef_ = r_models_functions.get_model_coefficients(self.model)
        self.sigma2 = r_models_functions.get_arima_sigma2(self.model)[0]
        self.fitted = r_models_functions.get_fitted_values(self.model)

        return self.model

    def predict(self, horizon_max: int, xreg_test: Union[np.ndarray, NULLType]) -> DataFrame:
        check_is_fitted(self, "is_fitted_")
        forecast = r_models_functions.predict_arima(self.model, horizon_max, xreg_test)
        try:
            forecast["lower"] = forecast["lower"].clip(lower=0)
        except ValueError: # TODO why?
            forecast = DataFrame(forecast)
        return forecast


class ExponentialSmoothing(TimeSeriesModel):
    def __init__(self, error_type: str = 'Z', trend_type: str = 'Z', season_type: str = 'Z', damped=NULL):
        super().__init__()
        self.error_type = error_type
        self.trend_type = trend_type
        self.season_type = season_type
        self.damped = damped

    def fit(self, y_train: np.ndarray, xreg_train: NULLType) -> TimeSeriesModel:
        if self.is_fitted_:
            self.model = r_models_functions.refit_ets(self.model, y_train)
        else:
            order_string = self.error_type + self.trend_type + self.season_type
            self.model = r_models_functions.fit_ets(order_string, y_train, self.damped)
            self.is_fitted_ = True

        self.coef_ = r_models_functions.get_model_coefficients(self.model)
        self.sigma2 = r_models_functions.get_ets_sigma2(self.model)[0]
        self.fitted = r_models_functions.get_fitted_values(self.model)

        return self.model

    def predict(self, horizon_max: int, xreg_test: Union[np.ndarray, NULLType]) -> DataFrame:
        check_is_fitted(self, "is_fitted_")
        forecast = r_models_functions.predict_ets(self.model, horizon_max)
        try:
            forecast["lower"] = forecast["lower"].clip(lower=0)
        except ValueError: # TODO why?
            forecast = DataFrame(forecast)
        return forecast

class SExpS(TimeSeriesModel):
    def fit_predict(
        self,
        y_train: np.ndarray,
        y_test: pd.Series,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            y_hat = r_models_functions.sexps(y_train, y_test.shape[0])
        except Exception:
            raise ModelNotFitError
        df_forecast = DataFrame(y_hat, columns=[y_test.name], index=y_test.index)
        return df_forecast

class Croston(TimeSeriesModel):
    def __init__(self, type='classic'):
        super().__init__()
        self.type = type

    def fit_predict(
        self,
        y_train: np.ndarray,
        y_test: pd.Series,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            y_hat = r_models_functions.croston(y_train, y_test.shape[0], self.type)
        except Exception:
            raise ModelNotFitError
        df_forecast = DataFrame(y_hat, columns=[y_test.name], index=y_test.index)
        return df_forecast

class TSB(TimeSeriesModel):
    def fit_predict(
        self,
        y_train: np.ndarray,
        y_test: pd.Series,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            y_hat = r_models_functions.tsb(y_train, y_test.shape[0])
        except Exception:
            raise ModelNotFitError
        df_forecast = DataFrame(y_hat, columns=[y_test.name], index=y_test.index)
        return df_forecast

class ADIDA(TimeSeriesModel):
    def fit_predict(
        self,
        y_train: np.ndarray,
        y_test: pd.Series,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            y_hat = r_models_functions.adida(y_train, y_test.shape[0])
        except Exception:
            raise ModelNotFitError
        df_forecast = DataFrame(y_hat, columns=[y_test.name], index=y_test.index)
        return df_forecast

class iMAPA(TimeSeriesModel):
    def fit_predict(
        self,
        y_train: np.ndarray,
        y_test: pd.Series,
        xreg_train: Union[np.ndarray, NULLType],
        xreg_test: Union[np.ndarray, NULLType],
    ) -> DataFrame:
        try:
            y_hat = r_models.imapa(y_train, y_test.shape[0])
        except Exception:
            raise ModelNotFitError
        df_forecast = DataFrame(y_hat, columns=[y_test.name], index=y_test.index)
        return df_forecast

class OES(TimeSeriesModel):
    def fit(self, y_train: np.ndarray, xreg_train: np.ndarray) -> TimeSeriesModel:
        self.model = r_models_functions.fit_oes(y_train, xreg_train)
        self.is_fitted_ = True
        self.fitted = r_models_functions.get_fitted_values(self.model)
        return self.model

    def predict(self, horizon_max: int, xreg_test: np.ndarray) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        forecast = r_models_functions.predict_oes(self.model, horizon_max, xreg_test)
        forecast = pd.DataFrame(forecast['Point.Forecast'], columns=['prediction'])
        return forecast

class ESX(TimeSeriesModel):
    def fit(self, y_train: np.ndarray, xreg_train: np.ndarray) -> TimeSeriesModel:
        self.model = r_models_functions.fit_esx(y_train, xreg_train)
        self.is_fitted_ = True
        self.fitted = r_models_functions.get_fitted_values(self.model)
        return self.model

    def predict(self, horizon_max: int, xreg_test: np.ndarray) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")
        forecast = r_models_functions.predict_esx(self.model, horizon_max, xreg_test)
        forecast = pd.DataFrame(forecast['Point.Forecast'], columns=['prediction'])
        return forecast
