% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\alias{bsts.prediction.errors}
\name{bsts.prediction.errors}
\title{One step prediction errors}
\Rdversion{1.1}
\description{
  Computes the one-step-ahead prediction errors for a model of class \code{\link{bsts}}.
}

\usage{
bsts.prediction.errors(bsts.object,
                       burn = SuggestBurn(.1, bsts.object))
}

\arguments{

  \item{bsts.object}{ An object of class \code{\link{bsts}}.}

  \item{burn}{An integer giving the number of MCMC iterations to discard
    as burn-in.  If \code{burn <= 0} then no burn-in sample will be
    discarded.}
}

\value{
  A matrix of draws of the one-step-ahead prediction errors.  Rows of
  the matrix correspond to MCMC draws.  Columns correspond to time.
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
  \code{\link[BoomSpikeSlab]{SpikeSlabPrior}},
  \code{\link[Boom]{SdPrior}}.
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  errors <- bsts.prediction.errors(model, burn = 100)
  PlotDynamicDistribution(errors)
}
\keyword{models}
\keyword{regression}
