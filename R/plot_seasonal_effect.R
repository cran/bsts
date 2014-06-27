PlotSeasonalEffect <- function(model,
                               nseasons = 7,
                               season.duration = 1,
                               same.scale = TRUE,
                               ylim = NULL,
                               getname = NULL,
                               ...) {
  ## Creates a set of plots similar to a 'month plot' showing how the
  ## effect of each season has changed over time.  This function uses
  ## mfrow to create a page of plots, so it cannot be used on the same
  ## page as other plotting functions.
  ##
  ## Args:
  ##   model:  A bsts model containing a seasonal component.
  ##   ylim:  The limits of the vertical axis.
  ##   same.scale: Used only if ylim is NULL.  If TRUE then all
  ##     figures are plotted on the same scale.  If FALSE then each
  ##     figure is independently scaled.
  ##   nseasons:  The number of seasons in the seasonal component to be plotted.
  ##   season.duration: The duration of each season in the seasonal
  ##     component to be plotted.
  ##   getname: A function taking a Date, POSIXt, or other time object
  ##     used as the index of the original data series used to fit
  ##     'model,' and returning a character string that can be used as
  ##     a title for each panel of the plot.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ##
  ## Returns:
  ##   Invisible NULL.
  ##
  effect.names <- dimnames(model$state.contributions)$component
  position <- grep("seasonal", effect.names)
  if (length(position) == 1) {
    name.components <- strsplit(effect.names[position], ".", fixed = TRUE)[[1]]
    nseasons <- as.numeric(name.components[2])
    season.duration <- as.numeric(name.components[3])
  } else {
    effect.name <- paste("seasonal", nseasons, season.duration, sep = ".")
    position <- grep(effect.name, effect.names)
  }
  if (length(position) != 1) {
    stop("The desired seasonal effect could not be located.")
  }
  effect <- model$state.contributions[, position, ]
  if (is.null(ylim) && same.scale == TRUE) {
    ylim <- range(effect)
  }
  vary.ylim <- is.null(ylim)
  time <- index(model$original.series)
  nr <- floor(sqrt(nseasons))
  nc <- ceiling(nseasons / nr)
  if (is.null(getname) && inherits(time, c("Date", "POSIXt"))) {
    if (nseasons == 7) {
      getname <- weekdays
    } else if (nseasons == 12) {
      getname <- months
    } else if (nseasons == 4) {
      getname <- quarters
    }
  }

  par(mfrow = c(nr, nc))
  for (season in 1:nseasons) {
    time.index <- seq(from = season,
                      to = length(time),
                      by = nseasons * season.duration)
    season.effect <- effect[, time.index]
    if (vary.ylim) {
      ylim <- range(season.effect)
    }
    dates <- time[time.index]
    PlotDynamicDistribution(season.effect, dates, ylim = ylim, ...)
    if (inherits(dates, c("Date", "POSIXt")) && !is.null(getname)) {
      season.name <- getname(dates[1])
      title(main = season.name)
    } else {
      title(main = paste("Season", season))
    }
  }
  return(invisible(NULL))
}
