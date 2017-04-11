### Functions for obtaining diagnostics (mainly different flavors of
### residuals) from bsts objects.
### ----------------------------------------------------------------------
residuals.bsts <- function(object,
                           burn = SuggestBurn(.1, object),
                           mean.only = FALSE,
                           ...) {
  ## Args:
  ##   object:  An object of class 'bsts'.
  ##   burn:  The number of iterations to discard as burn-in.
  ##   mean.only: Logical.  If TRUE then the mean residual for each
  ##     time period is returned.  If FALSE then the full posterior
  ##     distribution is returned.
  ##   ...: Not used.  This argument is here to comply with the
  ##     generic 'residuals' function.
  ##
  ## Returns:
  ##   If mean.only is TRUE then this function returns a vector of
  ##   residuals with the same "time stamp" as the original series.
  ##   If mean.only is FALSE then the posterior distribution of the
  ##   residuals is returned instead, as a matrix of draws.  Each row
  ##   of the matrix is an MCMC draw, and each column is a time point.
  ##   The colnames of the returned matrix will be the timestamps of
  ##   the original series, as text.
  if (object$family %in% c("logit", "poisson")) {
    stop("Residuals are not supported for Poisson or logit models.")
  }
  state <- object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
  if (!object$timestamp.info$timestamps.are.trivial) {
    state <- state[, object$timestamp.info$timestamp.mapping, drop = FALSE]
  }
  residuals <- t(t(state) - as.numeric(object$original.series))
  if (mean.only) {
    residuals <- zoo(colMeans(residuals), index(object$original.series))
  } else {
    residuals <- t(zoo(t(residuals), index(object$original.series)))
  }
  return(residuals)
}
###----------------------------------------------------------------------
bsts.prediction.errors <- function(bsts.object,
                                   burn = SuggestBurn(.1, bsts.object)) {
  ## Returns the posterior distribution of the one-step-ahead
  ## prediction errors from the bsts.object.  The errors are organized
  ## as a matrix, with rows corresponding to MCMC iteration, and
  ## columns corresponding to time.
  ## Args:
  ##   bsts.object:  An object created by a call to 'bsts'
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ## Returns:
  ##   A matrix of prediction errors, with rows corresponding to MCMC
  ##   iteration, and columns to time.
  stopifnot(is.numeric(burn),
            length(burn) == 1,
            burn < bsts.object$niter)
  if (bsts.object$family %in% c("logit", "poisson")) {
    stop("Prediction errors are not supported for Poisson or logit models.")
  }
  if (!is.null(bsts.object$one.step.prediction.errors)) {
    errors <- bsts.object$one.step.prediction.errors
    stopifnot(burn < nrow(errors))
    if (burn > 0) {
      errors <- errors[-(1:burn), , drop = FALSE]
    }
    if (!bsts.object$timestamp.info$timestamps.are.trivial) {
      errors <- errors[,
                       bsts.object$timestamp.info$timestamp.mapping,
                       drop = FALSE]
    }
    return(errors)
  }

  errors <- .Call("analysis_common_r_bsts_one_step_prediction_errors_",
                  bsts.object,
                  NULL,
                  burn,
                  PACKAGE = "bsts")
  return(errors)
}
