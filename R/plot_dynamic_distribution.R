# Copyright 2011 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

# This function is a modified version of a function called
# plot.dynamic.dist that stevescott wrote for his own use long ago.

PlotDynamicDistribution <-
  function(curves,
           time = 1:ncol(curves),
           quantile.step = .01,
           xlim = range(time),
           xlab = "time",
           ylim = range(curves),
           ylab = "distribution",
           add = FALSE,
           ...) {
    ## Plots pointwise probability distributions as they evolve over
    ## 'time'.
    ## Args:
    ##   curves: a matrix where each row represents a curve (e.g. a
    ##     simulation of a time series from a posterior distribution)
    ##     and columns represent time.  I.e. a long time series would
    ##     be a wide matrix.
    ##   time: An optional vector of time points that 'curves' will be
    ##     plotted against.  Good choices for 'time' are numeric, or
    ##     Date (as in as.Date).
    ##   quantile.step: Manages the number of polygons used to create
    ##     the plot.  Smaller values lead to more polygons, which is a
    ##     smoother visual effect, but more polygons take more time to
    ##     plot.
    ##   xlim: the x limits (x1, x2) of the plot.  Note that ‘x1 > x2’
    ##     is allowed and leads to a ‘reversed axis’.
    ##   xlab: Label for the horizontal axis.
    ##   ylim: the y limits (y1, y2) of the plot.  Note that ‘y1 > y2’
    ##     is allowed and leads to a ‘reversed axis’.
    ##   ylab: Label for the vertical axis.
    ##   add: Logical.  If true then add the plot to the current
    ##     plot.  Otherwise a fresh plot will be created.
    ##   ...:  Extra arguments to be passed to 'plot'.
    ## Returns:
    ##   There is no return value from this function.  It produces a
    ##     plot on the current graphics device.

    .FilledPlot <- function(time,
                            quantile.matrix,
                            poly.color,
                            add = FALSE,
                            xlab,
                            ylab,
                            ylim = range(quantile.matrix),
                            xlim = range(time),
                            ...) {
      ## This is a driver function to draw one of the nested polygons
      ## for PlotDynamicDistribution
      ylo <- quantile.matrix[, 1]
      yhi <- quantile.matrix[, 2]

      stopifnot(length(time) == nrow(quantile.matrix))

      if (any(yhi < ylo)) {
        warning("second column of quantile.matrix must be >= the first")
      }

      if (!add) {
        plot(time, ylo, ylim = ylim, xlab = xlab, ylab = ylab, type = "n", ...)
      }

      ## Create X and Y values of polygon points, then draw the
      ## polygon.
      X <- c(time, rev(time), time[1])
      Y <- c(ylo, rev(yhi), ylo[1])
      polygon(X, Y, border = NA, col = poly.color)
    }
    ##--------------------------------------

    qtl <- seq(0, 1, by = quantile.step)
    ## quantile.matrix is the actual matrix of quantiles that are used
    ## to draw the curve polygons
    quantile.matrix <- t(apply(curves, 2, quantile, probs = qtl))

    nc <- ncol(quantile.matrix)
    number.of.quantile.steps <- (nc + 1) / 2
    if (number.of.quantile.steps < 3) {
      stop("'quantile.step' is too large in PlotDynamicDistribution")
    }

    lower.quantile <- quantile.matrix[, 1]
    upper.quantile <- quantile.matrix[, nc]
    .FilledPlot(time,
                cbind(lower.quantile, upper.quantile),
                poly.color = gray(1 - (1 / number.of.quantile.steps)),
                axes = FALSE,
                add = add,
                xlim = xlim,
                xlab = xlab,
                ylim = ylim,
                ylab = ylab,
                ...)
    box()
    if (inherits(time, "Date")) {
      axis.Date(1, time, xpd = NA)
    } else if (inherits(time, "POSIXt")) {
      axis.POSIXct(1, as.POSIXct(time), xpd = NA)
    } else {
      axis(1, xpd = NA)
    }
    axis(2, xpd = NA)

    for (i in 2:(number.of.quantile.steps - 1)) {
      lower.quantile <- quantile.matrix[, i]
      upper.quantile <- quantile.matrix[, nc + 1 - i]
      .FilledPlot(time,
                  cbind(lower.quantile, upper.quantile),
                  add = TRUE,
                  poly.color = gray((1 - (i / number.of.quantile.steps))))
    }
    return(NULL)
  }
