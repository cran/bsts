\name{plot.seasonal.effect}
\title{Plot bsts seasonal effects}

\alias{PlotSeasonalEffect}

\description{Plot the seasonal effect from a \code{\link{bsts}} model.}

\usage{
PlotSeasonalEffect(model, nseasons = 7, season.duration = 1,
                   same.scale = TRUE, ylim = NULL, getname = NULL, ...)
}


\arguments{
  \item{model}{A model fit by \code{\link{bsts}}, containing a seasonal component.}

  \item{nseasons}{If there is only one seasonal component in the model,
    this argument is ignored.  If there are multiple seasonal
    components then \code{nseasons} and \code{season.duration} are used
    to select the desired one.}

  \item{season.duration}{If there is only one seasonal component in the
    model, this argument is ignored.  If there are multiple seasonal
    components then \code{nseasons} and \code{season.duration} are used
    to select the desired one.}

  \item{same.scale}{A logical indicating how to scale the vertical axis
    if \code{ylim} is unspecified.  If \code{TRUE} then all panels will
    have the same scale.  If \code{FALSE} then each panel is scaled
    independently.  If \code{ylim} is specified then this argument is
    ignored.}

  \item{ylim}{A length 2 vector giving the limits of the vertical axis.}

  \item{getname}{A function taking a single \code{\link{POSIXt}},
    \code{\link{Date}}, or similar object as an argument, and returning
    a single string that can be used as a panel title.}

  \item{\dots}{Extra arguments passed to
    \code{\link[Boom]{plot.dynamic.distribution}}.}

}

\value{
  Returns \code{invisible{NULL}}.
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  PlotSeasonalEffect(model)
}

\seealso{
  \code{\link{bsts}}
  \code{\link{PlotDynamicDistribution}}
}
