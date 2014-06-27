% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{add.generalized.local.linear.trend}
\alias{AddGeneralizedLocalLinearTrend}

\Rdversion{1.1}
\title{
  A generalized local linear trend state component
}

\description{
    The generalized local linear trend model is more complicated.
    It assumes the level moves according to a random walk, but the slope
    moves according to an AR1 process centered on some potentially
    nonzero value \eqn{D}{D}.  The equation for the mean is

    \deqn{\mu_{t+1} = \mu_t + \delta_t + \epsilon_t \qquad \epsilon_t \sim
      \mathcal{N(0, \sigma_\mu)}.}{ %
      mu[t+1] = mu[t] + delta[t] + rnorm(1, 0, sigma.level)
    }

    The equation for the slope is \deqn{\delta_{t+1} = D + \phi
    (\delta_t - D) + \eta_t \qquad \eta_t \sim \mathcal{N(0,
      \sigma_\delta)}.}{%
    delta[t+1] = D + phi * (delta[t] - D) + rnorm(1, 0, sigma.slope).}

    The prior distribution for this model has four independent components.
    There is an inverse gamma prior on the level standard deviation
    \eqn{\sigma_\mu}{sigma.level}, an inverse gamma prior on the slope
    standard deviation \eqn{\sigma_\delta}{sigma.slope}, a Gaussian
    prior on the long run slope parameter \eqn{D}{D}, and a potentially
    truncated Gaussian prior on the AR1 coefficient \eqn{\phi}{phi}.  If
    the prior on \eqn{\phi}{phi} is truncated to (-1, 1), then the slope
    will exhibit short term stationary variation around the long run
    slope \eqn{D}{D}.  }

\usage{
  AddGeneralizedLocalLinearTrend(
     state.specification,
     y,
     level.sigma.prior,
     slope.mean.prior,
     slope.ar1.prior,
     slope.sigma.prior,
     initial.level.prior,
     initial.slope.prior,
     sdy,
     initial.y)
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{The time series to be modeled, as a numeric vector.}

 \item{level.sigma.prior}{An object created by
   \code{\link[Boom]{SdPrior}} describing the prior distribution for
   the standard deviation of the level component.}

  \item{slope.mean.prior}{An object created by
   \code{\link[Boom]{NormalPrior}} giving the prior distribution for
   the mean parameter in the generalized local linear trend model (see
   below). }

  \item{slope.ar1.prior}{An object created by
   \code{\link[Boom]{Ar1CoefficientPrior}} giving the prior
   distribution for the ar1 coefficient parameter in the generalized
   local linear trend model (see below). }

 \item{slope.sigma.prior}{An object created by
   \code{\link[Boom]{SdPrior}} describing the prior distribution of
   the standard deviation of the slope component.}

  \item{initial.level.prior}{An object created by
    \code{\link[Boom]{NormalPrior}} describing the initial distribution
    of the level portion of the initial state vector.}

  \item{initial.slope.prior}{An object created by
    \code{\link[Boom]{NormalPrior}} describing the prior distribution
    for the slope portion of the initial state vector.}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

  \item{initial.y}{The initial value of the series being modeled.  This will be
    ignored if \code{y} is provided, or if the priors for the initial
    state are all provided directly.}
}

\value{ Returns a list with the elements necessary to specify a
  generalized local linear trend state model.}

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
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
}

\keyword{models}
