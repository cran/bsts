% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\alias{bsts}
\name{bsts}
\title{Bayesian structural time series}
\Rdversion{1.1}
\description{

  Uses MCMC to sample from the posterior distribution of a Bayesian
  structural time series model.  This function can be used either with
  or without contemporaneous predictor variables (in a time series
  regression).

  If predictor variables are present, the regression coefficients are
  fixed (as opposed to time varying, though time varying coefficients
  might be added as state component).  The predictors and response in
  the formula are contemporaneous, so if you want lags and differences
  you need to put them in the predictor matrix yourself.

  If no predictor variables are used, then the model is an ordinary
  state space time series model.
}

\usage{
bsts(formula,
     state.specification,
     save.state.contributions = TRUE,
     save.prediction.errors = TRUE,
     data,
     prior,
     contrasts = NULL,
     na.action = na.pass,
     niter,
     ping = niter / 10,
     seed = NULL,
     ...)
}

\arguments{

  \item{formula}{ A formula describing the regression portion of the
    relationship between y and X.

    If no regressors are desired then the formula can be replaced by a
    numeric vector giving the time series to be modeled.  Missing values
    are not allowed.

    If the response variable is of class \code{\link[zoo]{zoo}},
    \code{\link[xts]{xts}}, or \code{\link{ts}}, then the time series
    information it contains will be used in many of the plotting methods
    called from \code{\link{plot.bsts}}.  }

  \item{state.specification}{A list with elements created by
    \code{\link{AddLocalLinearTrend}}, \code{\link{AddSeasonal}}, and similar
    functions for adding components of state.  See the help page for
    \code{\link{state.specification}}.}

  \item{save.state.contributions}{Logical.  If \code{TRUE} then a 3-way
    array named \code{state.contributions} will be stored in the
    returned object.  The indices correspond to MCMC iteration, state
    model number, and time.  Setting \code{save.state.contributions} to
    \code{FALSE} yields a smaller object, but \code{plot} will not be
    able to plot the the "state", "components", or "residuals" for the
    fitted model.}

  \item{save.prediction.errors}{Logical.  If \code{TRUE} then a matrix
    named \code{one.step.prediction.errors} will be saved as part of the
    model object.  The rows of the matrix represent MCMC iterations, and
    the columns represent time.  The matrix entries are the
    one-step-ahead prediction errors from the Kalman filter.  }

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing the
    variables in the model.  If not found in \code{data}, the variables
    are taken from \code{environment(formula)}, typically the
    environment from which \code{\link{bsts}} is called.}

  \item{prior}{If regressors are supplied in the model formula, then
    this is a prior distribution for the regression component of the
    model, as created by \code{\link[BoomSpikeSlab]{SpikeSlabPrior}}.  The prior
    for the time series component of the model will be specified during
    the creation of state.specification.  This argument is only used if
    a formula is specified.

    If the model contains no regressors, then this is simply the prior
    on the residual standard deviation, expressed as an object created
    by \code{\link[Boom]{SdPrior}}. }

  \item{contrasts}{An optional list. See the \code{contrasts.arg} of
    \code{\link{model.matrix.default}}.  This argument is only used if a model
    formula is specified.  Even then the default is usually what you
    want.}

  \item{na.action}{What to do about missing values.  The default is to
    allow missing responses, but no missing predictors.  Set this to
    na.omit or na.exclude if you want to omit missing responses
    altogether.}

  \item{niter}{A positive integer giving the desired number of MCMC
    draws.}

  \item{ping}{ A scalar giving the desired frequency of status messages.
    If ping > 0 then the program will print a status message to the
    screen every \code{ping} MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{...}{ Extra arguments to be passed to
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}} (see the entry for the \code{prior}
    argument, above).}

}

\value{
  An object of class \code{\link{bsts}} which is a list with the
  following components

  \item{coefficients}{ A \code{niter} by \code{ncol(X)} matrix of MCMC
    draws of the regression coefficients, where \code{X} is the design
    matrix implied by \code{formula}.  This is only present if a model
    formula was supplied.}

  \item{sigma.obs}{A vector of length \code{niter} containing MCMC draws
    of the residual standard deviation.}

  The returned object will also contain named elements holding the MCMC
  draws of model parameters belonging to the state models.  The names of
  each component are supplied by the entries in
  \code{state.specification}.  If a model parameter is a scalar, then
  the list element is a vector with \code{niter} elements.  If the
  parameter is a vector then the list element is a matrix with
  \code{niter} rows.  If the parameter is a matrix then the list element
  is a 3-way array with first dimension \code{niter}.

  Finally, if a model formula was supplied, then the returned object
  will contain the information necessary for the predict method to build
  the design matrix when a new prediction is made.
}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts}},
  \code{\link{AddLocalLevel}},
  \code{\link{AddLocalLinearTrend}},
  \code{\link{AddGeneralizedLocalLinearTrend}},
  \code{\link{AddSeasonal}}
  \code{\link{AddDynamicRegression}}
  \code{\link[BoomSpikeSlab]{SpikeSlabPrior}},
  \code{\link[Boom]{SdPrior}}.
}

\examples{
  ## Example 1:  Time series (ts) data
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  pred <- predict(model, horizon = 12, burn = 100)
  par(mfrow = c(1,2))
  plot(model)
  plot(pred)

\dontrun{

  MakePlots <- function(model, ask = TRUE) {
    ## Make all the plots callable by plot.bsts.
    opar <- par(ask = ask)
    on.exit(par(opar))
    plot.types <- c("state", "components", "residuals",
                    "prediction.errors", "forecast.distribution")
    for (plot.type in plot.types) {
      plot(model, plot.type)
    }
    if (model$has.regression) {
      regression.plot.types <- c("coefficients", "predictors", "size")
      for (plot.type in regression.plot.types) {
        plot(model, plot.type)
      }
    }
  }

  ## Example 2: GOOG is the Google stock price, an xts series of daily
  ##            data.
  data(goog)
  ss <- AddGeneralizedLocalLinearTrend(list(), goog)
  model <- bsts(goog, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 3:  Change GOOG to be zoo, and not xts.
  goog <- zoo(goog, index(goog))
  ss <- AddGeneralizedLocalLinearTrend(list(), goog)
  model <- bsts(goog, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 4:  Naked numeric data works too
  y <- rnorm(100)
  ss <- AddLocalLinearTrend(list(), y)
  model <- bsts(y, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 5:  zoo data with intra-day measurements
  y <- zoo(rnorm(100),
           seq(from = as.POSIXct("2012-01-01 7:00 EST"), len = 100, by = 100))
  ss <- AddLocalLinearTrend(list(), y)
  model <- bsts(y, state.specification = ss, niter = 500)
  MakePlots(model)
}

}
\keyword{models}
\keyword{regression}
