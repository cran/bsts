# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AddDynamicRegression <- function(state.specification,
                                 formula,
                                 data,
                                 sigma.mean.prior = NULL,
                                 shrinkage.parameter.prior = NULL,
                                 contrasts = NULL,
                                 na.action = na.pass) {
  ## Args:
  ##   state.specification: a list with elements created by
  ##     AddLocalLinearTrend, AddSeasonal, and similar functions for
  ##     adding components of state.
  ##   formula: A formula describing the regression portion of the
  ##     relationship between y and X. If no regressors are desired
  ##     then the formula can be replaced by a numeric vector giving
  ##     the time series to be modeled.  Missing values are not
  ##     allowed.
  ##   data: an optional data frame, list or environment (or object
  ##     coercible by ‘as.data.frame’ to a data frame) containing the
  ##     variables in the model.  If not found in ‘data’, the
  ##     variables are taken from ‘environment(formula)’, typically
  ##     the environment from which ‘bsts’ is called.
  ##   contrasts: an optional list. See the ‘contrasts.arg’ of
  ##     ‘model.matrix.default’.  This argument is only used if a
  ##     model formula is specified.  It can usually be ignored even
  ##     then.
  ##   na.action: What to do about missing values.  The default is to
  ##     allow missing responses, but no missing predictors.  Set this
  ##     to na.omit or na.exclude if you want to omit missing
  ##     responses altogether.
  ##
  ## The model is
  ##
  ##      beta[i, t] ~ N(beta[i, t-1], sigsq[i] / variance_x[i])
  ##  1.0 / sigsq[i] ~ Gamma(a / b)
  ##
  ## That is, each coefficient has its own variance term, which is
  ## scaled by the variance of the i'th column of X.  The parameters
  ## of the hyperprior are interpretable as: sqrt(b/a) typical amount
  ## that a coefficient might change in a single time period, and 'a'
  ## is the 'sample size' or 'shrinkage parameter' measuring the
  ## degree of similarity in sigma[i] among the arms.
  ##
  ## In most cases we hope b/a is small, so that sigma[i]'s will be
  ## small and the series will be forecastable.  We also hope that 'a'
  ## is large because it means that the sigma[i]'s will be similar to
  ## one another.
  ##
  ## The default prior distribution is a pair of independent Gamma
  ## priors for sqrt(b/a) and a.  The mean of sigma[i] is set to .01 *
  ## sd(y) with shape parameter equal to 1.  The mean of the shrinkage
  ## parameter is set to 10, but with shape parameter equal to 1.
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))

  function.call <- match.call()
  my.model.frame <- match.call(expand.dots = FALSE)
  frame.match <- match(c("formula", "data", "na.action"),
                       names(my.model.frame), 0L)
  my.model.frame <- my.model.frame[c(1L, frame.match)]
  my.model.frame$drop.unused.levels <- TRUE

  # In an ordinary regression model the default action for NA's is to
  # delete them.  This makes sense in ordinary regression models, but
  # is dangerous in time series, because it artificially shortens the
  # time between two data points.  If the user has not specified an
  # na.action function argument then we should use na.pass as a
  # default, so that NA's are passed through to the underlying C++
  # code.
  if (! "na.action" %in% names(my.model.frame)) {
    my.model.frame$na.action <- na.pass
  }
  my.model.frame[[1L]] <- as.name("model.frame")
  my.model.frame <- eval(my.model.frame, parent.frame())
  model.terms <- attr(my.model.frame, "terms")
  y <- model.response(my.model.frame, "numeric")
  x <- model.matrix(model.terms, my.model.frame, contrasts)
  if (all(x[, 1] == 1)) {
    ## TODO(stevescott):  find a better test
    x <- x[, -1, drop = FALSE]
  }

  stopifnot(ncol(x) >= 1)
  stopifnot(nrow(x) >= 1)

  # TODO(stevescott):  Do you want to ensure that x and y conform?
  # TODO(stevescott):  handle missing data

  if (is.null(sigma.mean.prior)) {
    sigma.mean.prior <-
      GammaPrior(prior.mean = sd(as.numeric(y), na.rm = TRUE) * .01,
                 a = 1)
  }
  stopifnot(inherits(sigma.mean.prior, "DoubleModel"))

  if (is.null(shrinkage.parameter.prior)) {
    shrinkage.parameter.prior <- GammaPrior(a = 10, b = 1)
  }
  stopifnot(inherits(shrinkage.parameter.prior, "DoubleModel"))

  state.component <- list(name = "dynamic",
                          design.matrix = x,
                          sigma.mean.prior = sigma.mean.prior,
                          shrinkage.parameter.prior = shrinkage.parameter.prior,
                          size = ncol(x))
  class(state.component) <- c("DynamicRegression", "StateModel")

  state.specification[[length(state.specification) + 1]] <- state.component
  return(state.specification)
}
