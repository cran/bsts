% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{PlotDynamicDistribution}
\Rdversion{1.1}
\alias{PlotDynamicDistribution}
\title{
  Plots the evolution of a distribution over time.
}
\description{
  Produces an dynamic distribution plot where gray scale shading is used
  to show the evolution of a distribution over time.  This function is
  useful when there are too many time points to do side-by-side boxplots.
}
\usage{
PlotDynamicDistribution(curves, time = 1:ncol(curves), quantile.step=.01,
                        xlim = range(time), xlab = "time",
                        ylim = range(curves), ylab = "distribution",
                        add=FALSE, ...)
}
\arguments{

  \item{curves}{ A matrix where each row represents a curve (e.g. a
    simulation of a time series from a posterior distribution) and
    columns represent time.  A long time series would be a wide matrix.
    }

  \item{time}{ An optional vector of time points that 'curves' will be
     plotted against.  Good choices for \code{time} are numeric, or Date (see
     \code{\link{as.Date}}).
  }

  \item{quantile.step}{Each color step in the plot corresponds to this
    difference in quantiles..  Smaller values make prettier plots, but
    the plots take longer to produce.}

  \item{xlim}{The x limits (x1, x2) of the plot.  Note that \code{ x1 >
    x2 } is allowed and leads to a "reversed axis".}

  \item{xlab}{Label for the horzontal axis. }

  \item{ylim}{The y limits (y1, y2) of the plot.  Note that \code{ y1 >
    y2 } is allowed and leads to a "reversed axis".}

  \item{ylab}{Label for the vertical axis. }

  \item{add}{Logical.  If true then add the plot to the current plot.
    Otherwise a fresh plot will be created.}

  \item{\dots}{Extra arguments to pass on to \code{\link{plot}} }

}

\details{ The function works by passing many calls to
  \code{\link{polygon}}.  Each polygon is associated with a quantile
  level, with darker shading near the median. }

\value{ This function is called for its side effect, which is to produce
  a plot on the current graphics device.  }

\author{
  Steven L. Scott \email{stevescott@google.com}
}
\examples{
  x <- t(matrix(rnorm(1000 * 100, 1:100, 1:100), nrow=100))
  ## x has 1000 rows, and 100 columns.  Column i is N(i, i^2) noise.

  PlotDynamicDistribution(x)
  time <- as.Date("2010-01-01", format = "\%Y-\%m-\%d") + (0:99 - 50)*7
  PlotDynamicDistribution(x, time)

}

\keyword{ hplot }
