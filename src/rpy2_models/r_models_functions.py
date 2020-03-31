from rpy2.robjects import r
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)
# rpy2_logger.addFilter(lambda record: 'Model is being refit with current smoothing parameters but initial states are being re-estimated' not in record.msg)
r("library(forecast)")

auto_arima = r(
    """
    function(y, xreg, seasonal){
        ts(y) %>% auto.arima(max.p=10, max.q=3, max.order=10, xreg=xreg, seasonal=seasonal, lambda='auto', stepwise=F, approximation=F)
    }
"""
)

get_arima_order = r("function(model){ model %>% arimaorder()}")

get_arima_lambda = r("function(model){ model$lambda}")

fit_arima = r(
    """
            function(order, y_train, X_train){
                ts(y_train) %>%
                    Arima(order=order, xreg=X_train)
            }
        """
)

refit_arima = r(
    """
            function(model, y_train, X_train){
                ts(y_train) %>%
                    Arima(model=model, xreg=X_train)
            }
        """
)

predict_arima = r(
    """
            function(model, h, xreg=NULL){
                forecast(model, h=h, level=75, xreg=xreg) %>%
                    data.frame() %>%
                        setNames(c("prediction", "lower", "upper"))
            }
        """
)

auto_ets = r("function(y){ts(y) %>% ets(lambda=1)}")

fit_ets = r(
    """
            function(model, y_train, damped){
                ts(y_train) %>%
                    ets(model=model, damped=damped, lambda=1)
            }
        """
)

refit_ets = r(
    """
            function(model, y_train){
                ts(y_train) %>%
                    ets(model=model)
            }
        """
)

predict_ets = r(
    """
            function(model, h){
                forecast(model, h=h, level=75) %>%
                    data.frame() %>%
                        setNames(c("prediction", "lower", "upper"))
            }
        """
)

fit_meanf = r(
    """
            function(y_train, order){{
                tail(y_train, order) %>%
                    ts() %>%
                        meanf(level=75)
            }}
        """
)

predict_meanf = r(
    """
            function(model, h){
                forecast(model, h=h) %>%
                    as.data.frame() %>%
                        setNames(c("prediction", "lower", "upper"))
            }
        """
)

get_model_coefficients = r(
    """
    function(model) {
        coef(model) %>% data.frame()
    }
    """
)

get_fitted_values = r(
    """
    function(model) {
        fitted(model)
    }
"""
)

get_arima_sigma2 = get_ets_sigma2 = r(
    """
    function(model){
        model$sigma2
    }
"""
)

get_ma_sigma2 = r(
    """
    function(model){
        model$model$sd ^ 2
    }
"""
)

r(
"""
    library(zoo)


    intervals <- function(x){
    y<-c()
    k<-1
    counter<-0
    for (tmp in (1:length(x))){
        if(x[tmp]==0){
        counter<-counter+1
        }else{
        k<-k+1
        y[k]<-counter
        counter<-1
        }
    }
    y<-y[y>0]
    y[is.na(y)]<-1
    y
    }

    demand <- function(x){
    y<-x[x!=0]
    y
    }

    SES <- function(a, x, h, job){
    y <- c()
    y[1] <- x[1] #initialization

    for (t in 1:(length(x))){
        y[t+1] <- a*x[t]+(1-a)*y[t]
    }

    fitted <- head(y,(length(y)-1))
    forecast <- rep(tail(y,1),h)
    if (job=="train"){
        return(mean((fitted - x)^2))
    }else if (job=="fit"){
        return(fitted)
    }else{
        return(list(fitted=fitted,mean=forecast))
    }
    }

    """
)


sexps = r("""
function(x, h){
  a <- optim(c(0), SES, x=x, h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
  y <- SES(a=a, x=x, h=1, job="forecast")$mean
  forecast <- rep(as.numeric(y), h)
  return(forecast)
}
""")

croston = r("""
function(x, h, type){
    if (type=="classic"){
        mult <- 1
        a1 = a2 <- 0.1
    }else if (type=="optimized"){
        mult <- 1
        a1 <- optim(c(0), SES, x=demand(x), h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
        a2 <- optim(c(0), SES, x=intervals(x), h=1, job="train", lower = 0.1, upper = 0.3, method = "L-BFGS-B")$par
    }else if (type=="sba"){
        mult <- 0.95
        a1 = a2 <- 0.1
    }
    yd <- SES(a=a1, x=demand(x), h=1, job="forecast")$mean
    yi <- SES(a=a2, x=intervals(x), h=1, job="forecast")$mean
    forecast <- rep(as.numeric(yd/yi), h)*mult
    return(forecast)
}
""")

tsb = r("""
function(x, h){
  n <- length(x)
  p <- as.numeric(x != 0)
  z <- x[x != 0]

  a <- c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.8)
  b <- c(0.01,0.02,0.03,0.05,0.1,0.2,0.3)
  MSE <- c() ; forecast <- NULL
  for (atemp in a){
    for (btemp in b){
      zfit <- vector("numeric", length(x))
      pfit <- vector("numeric", length(x))
      zfit[1] <- z[1] ; pfit[1] <- p[1]

      for (i in 2:n) {
        pfit[i] <- pfit[i-1] + atemp*(p[i]-pfit[i-1])
        if (p[i] == 0) {
          zfit[i] <- zfit[i-1]
        }else {
          zfit[i] <- zfit[i-1] + btemp*(x[i]-zfit[i-1])
        }
      }
      yfit <- pfit * zfit
      forecast[length(forecast)+1] <- list(rep(yfit[n], h))
      yfit <- c(NA, head(yfit, n-1))
      MSE <- c(MSE, mean((yfit-x)^2, na.rm = T) )
    }
  }
  return(forecast[[which.min(MSE)]])
}
""")

adida = r("""
function(x, h){
  al <- round(mean(intervals(x)),0) #mean inter-demand interval
  #Aggregated series (AS)
  AS <- as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al))))
  forecast <- rep(SexpS(AS, 1)/al, h)
  return(forecast)
}
""")

imapa = r("""
function(x, h){
  mal <- round(mean(intervals(x)),0)
  frc <- NULL
  for (al in 1:mal){
    frc <- rbind(frc, rep(SexpS(as.numeric(na.omit(as.numeric(rollapply(tail(x, (length(x) %/% al)*al), al, FUN=sum, by = al)))), 1)/al, h))
  }
  forecast <- colMeans(frc)
  return(forecast)
}
""")

r("library(smooth)")

fit_oes = r("""
    function(y_train, xreg=NULL){
        ts(y_train, frequency=7) %>%
            oes(model="YYY", occurence="auto", xreg=xreg)
    }
""")

predict_oes = r("""
    function(model, h, xreg=NULL){
        forecast(model, h=h, xreg=xreg, interval="n") %>%
            data.frame()
    }
""")

r("library(smooth)")

fit_esx = r("""
    function(y_train, xreg){
        ts(y_train, frequency=7) %>%
            es(xreg, occurence="auto")
    }
""")

predict_esx = r("""
    function(model, h, xreg){
        forecast(model, h=h, xreg=xreg, interval='n') %>%
            data.frame()
    }
""")
