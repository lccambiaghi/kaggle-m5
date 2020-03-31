aggregate_sales <- function(dated_sales){
  dated_sales$date <- as.Date(dated_sales$date, format = "%y-%m-%d")

  sales_ts <- dated_sales %>%
    as_tsibble(index=date, key = c(item_id, dept_id, cat_id, store_id, state_id))
    ## fill_gaps(.full = TRUE)

  sales_agg <- sales_ts %>%
    aggregate_key((state_id / store_id / dept_id) * item_id, sales = sum(sales))
}

fit_model <- function(sales_agg, model_spec){
  ## model_spec <- ETS(sales ~ error("A") + trend("N") + season("N"), opt_crit = "mse")
  sales_fit <- sales_agg %>%
    ## filter_index("2016-01-01" ~ "2016-03-27")  %>%
    model(mod = model_spec)
}

forecast_reconciled <- function(sales_fit){
  sales_fc <- sales_fit  %>%
    reconcile(
      ## method = c("wls_var", "ols", "wls_struct", "mint_cov", "mint_shrink"),
      mod = min_trace(mod, method='mint_shrink')
    ) %>%
    forecast(h="28 days")
}
