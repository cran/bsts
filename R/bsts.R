# Copyright 2011 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

## Two use cases:
## bsts(y ~ formula, data = my.data, state.specification = ss)
## bsts(y, state.specification = ss)

## Then we can plot.bsts, predict.bsts, etc.  Of course, they will
## need different argument lists depending on the presence/absence of
## predictors.

###----------------------------------------------------------------------
bsts <- function(formula,
                 state.specification,
                 save.state.contributions = TRUE,
                 save.prediction.errors = TRUE,
                 data,
                 prior = NULL,
                 contrasts = NULL,
                 na.action = na.pass,
                 niter,
                 ping = niter / 10,
                 seed = NULL,
                 ...) {
  ## Uses MCMC to sample from the posterior distribution of a Bayesian
  ## structural time series model.  This function can be used either
  ## with or without contemporaneous predictor variables (in a time
  ## series regression).

  ## If predictor variables are present, the regression coefficients
  ## are fixed (as opposed to time varying, though time varying
  ## coefficients might be added as part of a state variable).  The
  ## predictors and response in the formula are contemporaneous, so if
  ## you want lags and differences you need to put them in the
  ## predictor matrix yourself.

  ## If no predictor variables are used, then the model is an ordinary
  ## state space time series model.

  ## Args:
  ##   formula: A formula describing the regression portion of the
  ##     relationship between y and X. If no regressors are desired
  ##     then the formula can be replaced by a numeric vector giving
  ##     the time series to be modeled.  Missing values are not
  ##     allowed.  If the response is of class zoo, xts, or ts then
  ##     time series information it contains will be used in many of
  ##     the plot methods called by plot.bsts.
  ##   state.specification: a list with elements created by
  ##     AddLocalLinearTrend, AddSeasonal, and similar functions for
  ##     adding components of state.
  ##   save.state.contributions: Logical.  If TRUE then a 3-way array
  ##     named 'state.contributions' will be stored in the returned
  ##     object.  The indices correspond to MCMC iteration, state
  ##     model number, and time.  Setting 'save.state.contributions'
  ##     to 'FALSE' yields a smaller object, but plot() will not be
  ##     able to plot the the "state", "components", or "residuals"
  ##     for the fitted model.
  ##   save.prediction.errors: Logical.  If TRUE then a matrix named
  ##     'one.step.prediction.errors' will be saved as part of the
  ##     model object.  The rows of the matrix represent MCMC
  ##     iterations, and the columns represent time.  The matrix
  ##     entries are the one-step-ahead prediction errors from the
  ##     Kalman filter.
  ##   data: an optional data frame, list or environment (or object
  ##     coercible by ‘as.data.frame’ to a data frame) containing the
  ##     variables in the model.  If not found in ‘data’, the
  ##     variables are taken from ‘environment(formula)’, typically
  ##     the environment from which ‘bsts’ is called.
  ##   prior: If a regression component is present then this a prior
  ##     distribution for the regression component of the model, as
  ##     created by SpikeSlabPrior.  The prior for the time series
  ##     component of the model will be specified during the creation
  ##     of state.specification.  If no regression components are
  ##     specified then this is a prior for the residual standard
  ##     deviation, created by SdPrior.  In either case the prior is
  ##     optional.  A weak default prior will be used if no prior is
  ##     specified explicitly.
  ##   contrasts: an optional list. See the ‘contrasts.arg’ of
  ##     ‘model.matrix.default’.  This argument is only used if a
  ##     model formula is specified.  It can usually be ignored even
  ##     then.
  ##   na.action: What to do about missing values.  The default is to
  ##     allow missing responses, but no missing predictors.  Set this
  ##     to na.omit or na.exclude if you want to omit missing
  ##     responses altogether.
  ##   niter: a positive integer giving the desired number of MCMC
  ##     draws
  ##   ping: A scalar.  If ping > 0 then the program will print a
  ##     status message to the screen every 'ping' MCMC iterations.
  ##   seed: An integer to use as the C++ random seed.  If NULL then
  ##     the C++ seed will be set using the clock.
  ##   ...:  Extra arguments to be passed to SpikeSlabPrior.
  ## Returns:
  ##   An object of class 'bsts', which is a list with the following components
  ##   coefficients: a 'niter' by 'ncol(X)' matrix of MCMC draws of
  ##     the regression coefficients, where 'X' is the design matrix
  ##     implied by 'formula'.  This is only present if a model
  ##     formula was supplied
  ##   sigma.obs: a vector of length 'niter' containing MCMC draws of the
  ##     residual standard deviation.
  ##
  ##   The returned object will also contain named elements holding
  ##   the MCMC draws of model parameters belonging to the state
  ##   models.  The names of each component are supplied by the
  ##   entries in state.specification.  If a model parameter is a
  ##   scalar, then the list element is a vector with 'niter'
  ##   elements.  If the parameter is a vector then the list element
  ##   is a matrix with 'niter' rows, and if the parameter is a matrix
  ##   then the list element is a 3-way array with first dimension
  ##   'niter'.
  ##
  ##   Finally, if a model formula was supplied, then the returned
  ##   object will contain the information necessary for the predict
  ##   method to build the design matrix when a new prediction is
  ##   made.

  if (is.numeric(formula)) {
    y <- formula
    regression <- FALSE
  } else {
    regression <- TRUE
  }

  stopifnot(is.null(seed) || length(seed) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  if (regression) {
    stopifnot(is.numeric(niter))
    function.call <- match.call()
    my.model.frame <- match.call(expand.dots = FALSE)
    frame.match <- match(c("formula", "data", "na.action"),
                         names(my.model.frame), 0L)
    my.model.frame <- my.model.frame[c(1L, frame.match)]
    my.model.frame$drop.unused.levels <- TRUE

    # In an ordinary regression model the default action for NA's is
    # to delete them.  This makes sense in ordinary regression models,
    # but is dangerous in time series, because it artificially
    # shortens the time between two data points.  If the user has not
    # specified an na.action function argument then we should use
    # na.pass as a default, so that NA's are passed through to the
    # underlying C++ code.
    if (! "na.action" %in% names(my.model.frame)) {
      my.model.frame$na.action <- na.pass
    }
    my.model.frame[[1L]] <- as.name("model.frame")
    my.model.frame <- eval(my.model.frame, parent.frame())
    model.terms <- attr(my.model.frame, "terms")

    y <- model.response(my.model.frame, "numeric")
    x <- model.matrix(model.terms, my.model.frame, contrasts)
    if (any(is.na(x))) {
      stop("bsts does not allow NA's in the predictors, only the responses.")
    }
    if (is.null(prior)) {
      zero <- rep(0, ncol(x))
      prior <- SpikeSlabPrior(x,
                              y,
                              optional.coefficient.estimate = zero,
                              ...)
    }
    if (is.null(prior$max.flips)) {
      prior$max.flips <- -1
    }
    stopifnot(inherits(prior, "SpikeSlabPrior"))

    ## Identify any columns that are all zero, and assign them zero
    ## prior probability of being included in the model.
    all.zero <- apply(x, 2, function(z) all(z == 0))
    prior$prior.inclusion.probabilities[all.zero] <- 0

    stopifnot(nrow(x) == length(y))
    ans <- .Call("bsts_fit_state_space_regression_",
                 as.matrix(x),
                 as.vector(y),
                 as.logical(!is.na(y)),
                 state.specification,
                 as.logical(save.state.contributions),
                 as.logical(save.prediction.errors),
                 prior,
                 niter,
                 ping,
                 seed,
                 PACKAGE = "bsts")

    ans$regression.prior <- prior

    ## The following will be needed by the predict and summary
    ## methods.
    ans$contrasts <- attr(x, "contrasts")
    ans$xlevels <- .getXlevels(model.terms, my.model.frame)
    ans$call <- function.call
    ans$terms <- model.terms
    ans$mf <- my.model.frame
    ans$has.regression <- TRUE
    ans$design.matrix <- x

    variable.names <- dimnames(x)[[2]]
    if (!is.null(variable.names)) {
        dimnames(ans$coefficients)[[2]] <- variable.names
    }

  } else {
    ## Handle the non-regression case.
    stopifnot(is.numeric(y))

    if (missing(prior)) {
      prior <- SdPrior(sigma.guess = sd(y, na.rm = TRUE), sample.size = .01)
    }
    stopifnot(inherits(prior, "SdPrior"))
    ans <- .Call("bsts_fit_state_space_model_",
                 y,
                 as.logical(!is.na(y)),
                 state.specification,
                 save.state.contributions,
                 save.prediction.errors,
                 niter,
                 prior,
                 ping,
                 seed,
                 PACKAGE = "bsts")
    ans$has.regression <- FALSE
  }
  ## End of main if-else block, so ans has been populated in both the
  ## regression and non-regression cases.

  ans$state.specification <- state.specification
  ## All the plotting functions depend on y being a zoo or object, so
  ## they can call index() on it to get the dates.  Note that a ts or
  ## plain numeric object can be converted to zoo using as.zoo.
  ans$original.series <- as.zoo(y)

  if (save.state.contributions) {
    ## Store the names of each state model in the appropriate dimname
    ## for state.contributions.
    nstate <- length(state.specification)
    state.names <- character(nstate)
    for (i in seq_len(nstate)) state.names[i] <- state.specification[[i]]$name
    if (ans$has.regression) {
      state.names <- c(state.names, "regression")
    }
    dimnames(ans$state.contributions) <- list(mcmc.iteration = NULL,
                                              component = state.names,
                                              time = NULL)
  }

  if (ans$has.regression) {
    ans <- .RemoveInterceptAmbiguity(ans)
  }

  class(ans) <- "bsts"
  return(ans)
}

.RemoveInterceptAmbiguity <- function(bsts.object) {
  ## If the model contains a regression with an intercept term then
  ## there is an indeterminacy between the intercept and the trend.
  ## We need to subtract the intercept term from the regression
  ## component and add it to the trend component.

  if (!bsts.object$has.regression) return(bsts.object)

  ## Compute a logical vector indicating which columns contain all 1's.
  all.ones <- apply(bsts.object$design.matrix,
                    2,
                    function(column)(all(column == 1)))
  state.sizes <- StateSizes(bsts.object$state.specification)
  has.trend <- "trend" %in% names(state.sizes)

  if (has.trend && any(all.ones)) {
    if (sum(all.ones) > 1) {
      warning("The predictor matrix contains multiple columns of 1's.  ",
              "Treating the first one as the intercept.")
    }
    intercept.position = match(TRUE, all.ones)
    intercept <- bsts.object$coefficients[, intercept.position]

    state.names <- dimnames(bsts.object)[[2]]
    trend.position <- match("trend", state.names)
    if (is.na(trend.position)) trend.position <- 1

    bsts.object$state.contributions[, trend.position, ] <-
      bsts.object$state.contributions[, trend.position, ] + intercept
    bsts.object$state.contributions[, "regression", ] <-
      bsts.object$state.contributions[, "regression", ] - intercept
    bsts.object$coefficients[, intercept.position] <- 0

    ## We also need to add the intercept term into the right component
    ## of "final state".  We need to find the right location again,
    ## because final.state (a full state vector) is indexed
    ## differently than state.contributions (which just gives the
    ## overall contribution of each state component).
    trend.components.index <- match("trend", names(state.sizes))
    trend.components.index <-
      trend.components.index[!is.na(trend.components.index)]
    stopifnot(length(trend.components.index) == 1)
    trend.starting.position <- 1
    if (trend.components.index > 1) {
      trend.starting.position <-
        1 + cumsum(state.sizes[1:(trend.components.index - 1)])
    }

    bsts.object$final.state[, trend.starting.position] <-
          bsts.object$final.state[, trend.starting.position] + intercept
  }
  return(bsts.object)
}

###----------------------------------------------------------------------
plot.bsts <- function(x,
                      y = c("state", "components", "residuals", "coefficients",
                        "prediction.errors", "forecast.distribution",
                        "predictors", "size",
                        "dynamic"),
                      ...) {
  ## S3 method for plotting bsts objects.
  ## Args:
  ##   x: An object of class 'bsts'.
  ##   y: character string indicating the aspect of the model that
  ##     should be plotted.  Partial matching is allowed,
  ##     so 'y = "res"' will produce a plot of the residuals.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.

  y <- match.arg(y)
  if (y == "state") {
    PlotBstsState(x, ...)
  } else if (y == "components") {
    PlotBstsComponents(x, ...)
  } else if (y == "residuals") {
    PlotBstsResiduals(x, ...)
  } else if (y == "coefficients") {
    PlotBstsCoefficients(x, ...)
  } else if (y == "prediction.errors") {
    PlotBstsPredictionErrors(x, ...)
  } else if (y == "forecast.distribution") {
    PlotBstsForecastDistribution(x, ...)
  } else if (y == "predictors") {
    PlotBstsPredictors(x, ...)
  } else if (y == "size") {
    PlotBstsSize(x, ...)
  } else if (y == "dynamic") {
    PlotDynamicRegression(x, ...)
  }
}

###----------------------------------------------------------------------
summary.bsts <- function(object, burn = SuggestBurn(.1, object), ...) {
  ## Prints a summary of the supplied bsts object.
  ## Args:
  ##   object:  An object of class 'bsts'
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   ...: Additional arguments passed to summary.lm.spike, if
  ##     'object' has a regression component.
  ## Returns:
  ##   A list of summaries describing the bsts object.
  ##   residual.sd: The posterior mean of the residual standard
  ##     deviation parameter.
  ##   prediction.sd: The standard deviation of the one-step-ahead
  ##     prediction errors.  These differ from the residuals because
  ##     they only condition on the data preceding the prediction.
  ##     The residuals condition on all data in both directions.
  ##   rquare: The R-square from the model, computing by comparing
  ##     'residual.sd' to the sample variance of the original series.
  ##   relative.gof: Harvey's goodness of fit statistic:
  ##     1 - SSE(prediction errors) / SST(first difference of original series).
  ##     This is loosly analogous to the R^2 in a regression model.
  ##     It differs in that the baseline model is a random walk with
  ##     drift (instead of the sample mean).  Models that fit worse,
  ##     on average, than the baseline model can have a negative
  ##     relative.gof score.
  ##   size: If the original object had a regression component, then
  ##     'size' summarizes the distribution of the number of nonzero
  ##     coefficients.
  ##   coefficients: If the original object had a regression
  ##     component, then 'coef' contains a summary of the regression
  ##     coefficients computed using summary.lm.spike.

  stopifnot(inherits(object, "bsts"))
  sigma.obs <- object$sigma.obs
  residual.sd <- mean(sigma.obs)
  rsquare <- 1 - residual.sd^2 / var(object$original.series)

  prediction.errors <- bsts.prediction.errors(object, burn = burn)
  prediction.sse <- sum(colMeans(prediction.errors)^2)
  original.series <- as.numeric(object$original.series)
  dy <- diff(original.series)
  prediction.sst <- var(dy) * (length(dy) - 1)

  ans <- list(residual.sd = residual.sd,
              prediction.sd = sd(colMeans(prediction.errors)),
              rsquare = rsquare,
              relative.gof = 1 - prediction.sse / prediction.sst)

  ##----------------------------------------------------------------------
  ## summarize the regression coefficients
  if (object$has.regression) {
    beta <- object$coefficients
    if (burn > 0) {
      beta <- beta[-(1:burn), , drop = FALSE]
    }
    include <- beta != 0
    model.size <- rowSums(include)
    ans$size <- summary(model.size)
    ans$coefficients <- SummarizeSpikeSlabCoefficients(object$coefficients,
                                                       burn = burn, ...)
  }
  return(ans)
}

###----------------------------------------------------------------------
PlotBstsPredictors <- function(bsts.object,
                               burn = SuggestBurn(.1, bsts.object),
                               inclusion.threshold = .10,
                               ylim = NULL,
                               flip.signs = TRUE,
                               show.legend = TRUE,
                               grayscale = TRUE,
                               short.names = TRUE,
                               ...) {
  ## Plots the time series of predictors with high probabilities of inclusion
  ## Args:
  ##   bsts.object:  A bsts object containing a regression component.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   inclusion.threshold: An inclusion probability that coefficients
  ##     must exceed in order to be displayed.
  ##   ylim:  Limits on the vertical axis.
  ##   flip.signs: If true then a predictor with a negative sign will
  ##     be flipped before being plotted, to better align visually
  ##     with the target series.
  ##   ...:  Extra arguments passed to either 'plot' or 'plot.zoo'.
  ## Returns:
  ##   Invisible NULL.
  stopifnot(inherits(bsts.object, "bsts"))
  beta <- bsts.object$coefficients
  if (burn > 0) {
    beta <- beta[-(1:burn), ]
  }

  inclusion.probabilities <- colMeans(beta != 0)
  keep <- inclusion.probabilities > inclusion.threshold
  if (any(keep)) {

    predictors <- bsts.object$design.matrix[, keep, drop = FALSE]
    predictors <- scale(predictors)
    if (flip.signs) {
      compute.positive.prob <- function(x) {
        x <- x[x != 0]
        if (length(x) == 0) {
          return(0)
        }
        return(mean(x > 0))
      }
      positive.prob <- apply(beta[, keep, drop = FALSE], 2,
                             compute.positive.prob)
      signs <- ifelse(positive.prob > .5, 1, -1)
      predictors <- scale(predictors, scale = signs)
    }

    inclusion.probabilities <- inclusion.probabilities[keep]
    number.of.predictors <- ncol(predictors)
    original <- scale(bsts.object$original.series)
    if (is.null(ylim)) {
      ylim <- range(predictors, original)
    }
    index <- rev(order(inclusion.probabilities))
    predictors <- predictors[, index]
    inclusion.probabilities <- inclusion.probabilities[index]
    predictor.names <- colnames(predictors)
    if (short.names) {
      predictor.names <- Shorten(predictor.names)
    }

    if (grayscale) {
      line.colors <- gray(1 - inclusion.probabilities)
    } else {
      line.colors <- rep("black", number.of.predictors)
    }
    times <- index(bsts.object$original.series)
    if (number.of.predictors == 1) {
      plot(times, predictors, type = "l", lty = 1, col = line.colors,
           ylim = ylim, xlab = "", ylab = "Scaled Value", ...)
    } else {
      plot(times, predictors[, 1], type = "n", ylim = ylim, xlab = "",
           ylab = "Scaled Value", ...)
      for (i in 1:number.of.predictors) {
        lines(times, predictors[, i], lty = i, col = line.colors[i], ...)
      }
    }
    lines(zoo(scale(bsts.object$original.series),
              index(bsts.object$original.series)),
          col = "blue",
          lwd = 3)
    if (show.legend) {
      legend.text <- paste(round(inclusion.probabilities, 2), predictor.names)
      legend("topright", legend = legend.text, lty = 1:number.of.predictors,
             col = line.colors, bg = "white")
    }
  } else {
    plot(0, 0, type = "n",
         main = "No predictors above the inclusion threshold.", ...)
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotBstsCoefficients <- function(bsts.object,
                                 burn = SuggestBurn(.1, bsts.object),
                                 inclusion.threshold = 0,
                                 number.of.variables = NULL,
                                 ...) {
  ## Creates a plot of the regression coefficients in the bsts.object.
  ## This is a wrapper for plot.lm.spike from the BoomSpikeSlab package.
  ## Args:
  ##   bsts.object:  An object of class 'bsts'
  ##   burn: The number of MCMC iterations to discard as burn-in.
  ##   inclusion.threshold: An inclusion probability that coefficients
  ##     must exceed in order to be displayed.
  ##   number.of.variables: If non-NULL this specifies the number of
  ##     coefficients to plot, taking precedence over
  ##     inclusion.threshold.
  ## Returns:
  ##   Invisibly returns a list with the following elements:
  ##   barplot: The midpoints of each bar, which is useful for adding
  ##     to the plot
  ##   inclusion.prob: The marginal inclusion probabilities of each
  ##     variable, ordered smallest to largest (the same ordering as
  ##     the plot).
  ##   positive.prob: The probability that each variable has a
  ##     positive coefficient, in the same order as inclusion.prob.
  ##   permutation: The permutation of beta that puts the coefficients
  ##     in the same order as positive.prob and inclusion.prob.  That
  ##     is: beta[, permutation] will have the most significant
  ##     coefficients in the right hand columns.
  stopifnot(inherits(bsts.object, "bsts"))
  if (is.null(bsts.object$coefficients)) {
    stop("no coefficients to plot in PlotBstsCoefficients")
  }
  coef <- bsts.object$coefficients
  class(coef) <- "lm.spike"
  tmp <- list(beta = bsts.object$coefficients)
  class(tmp) <- "lm.spike"
  return(invisible(plot.lm.spike(tmp,
                                 burn,
                                 inclusion.threshold,
                                 number.of.variables = number.of.variables,
                                 ...)))
}
###----------------------------------------------------------------------
PlotBstsSize <- function(bsts.object,
                         burn = SuggestBurn(.1, bsts.object),
                         style = c("histogram", "ts"),
                         ...) {
  ## Plots the distribution of the number of variables in the bsts model.
  ## Args:
  ##   bsts.object:  An object of class 'bsts' to plot.
  ##   burn: The number of MCMC iterations to discard as burn-in.
  ##   style:  The desired plot style.
  ##   ...:  Extra arguments passed to lower level plotting functions.
  ## Returns:
  ##   Nothing interesting.  Draws a plot on the current graphics device.
  beta <- bsts.object$coefficients
  if (is.null(beta)) {
    stop("The model has no coefficients")
  }
  if (burn > 0) {
    beta <- beta[-(1:burn), ]
  }
  size <- rowSums(beta != 0)
  style <- match.arg(style)
  if (style == "ts") {
    plot.ts(size, ...)
  } else if (style == "histogram") {
    hist(size, ...)
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotBstsComponents <- function(bsts.object,
                               burn = SuggestBurn(.1, bsts.object),
                               time,
                               same.scale = TRUE,
                               layout = c("square", "horizontal", "vertical"),
                               style = c("dynamic", "boxplot"),
                               ylim = NULL,
                               ...) {
  ## Plots the posterior distribution of each state model's
  ## contributions to the mean of the time series.
  ## Args:
  ##   bsts.object: An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   same.scale: Logical.  If TRUE then all plots will share a
  ##     common scale for the vertical axis.  Otherwise the veritcal
  ##     scales for each plot will be determined independently.
  ##   layout: A text string indicating whether the state components
  ##     plots should be laid out in a square (maximizing plot area),
  ##     vertically, or horizontally.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  style <- match.arg(style)

  if (missing(time)) {
    time <- index(bsts.object$original.series)
  }
  state <- bsts.object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  dims <- dim(state)
  number.of.components <- dims[2]

  layout <- match.arg(layout)
  if (layout == "square") {
    num.rows <- floor(sqrt(number.of.components))
    num.cols <- ceiling(number.of.components / num.rows)
  } else if (layout == "vertical") {
    num.rows <- number.of.components
    num.cols <- 1
  } else if (layout == "horizontal") {
    num.rows <- 1
    num.cols <- number.of.components
  }
  original.par <- par(mfrow = c(num.rows, num.cols))
  on.exit(par(original.par))

  names <- dimnames(state)[[2]]

  have.ylim <- !is.null(ylim)
  if (same.scale) {
    scale <- range(state)
  }
  for (component in 1:number.of.components) {
    if (!have.ylim) {
      ylim <- if (same.scale) scale else range(state[ , component, ])
    }
    if (style == "boxplot") {
      TimeSeriesBoxplot(state[, component, ],
                        time = time,
                        ylim = ylim,
                        ...)
    } else {
      PlotDynamicDistribution(state[, component, ],
                              time = time,
                              ylim = ylim,
                              ...)
    }
    title(main = names[component])
  }
}

###----------------------------------------------------------------------
PlotBstsState <- function(bsts.object, burn = SuggestBurn(.1, bsts.object),
                          time, show.actuals = TRUE,
                          style = c("dynamic", "boxplot"), ...) {
  ## Plots the posterior distribution of the mean of the training
  ## data, as determined by the state.
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   show.actuals: If TRUE then the original values from the series
  ##     will be added to the plot.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...: Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  style <- match.arg(style)
  if (missing(time)) {
    time <- index(bsts.object$original.series)
  }
  state <- bsts.object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
  if (style == "boxplot") {
    TimeSeriesBoxplot(state, time = time, ...)
  } else {
    PlotDynamicDistribution(state, time = time, ...)
  }
  if (show.actuals) {
    points(time, bsts.object$original.series, col = "blue", ...)
  }
}

###----------------------------------------------------------------------
PlotBstsPredictionErrors <- function(bsts.object,
                                     burn = SuggestBurn(.1, bsts.object),
                                     time, style = c("dynamic", "boxplot"),
                                     ...) {
  ## Creates a dynamic distribution plot of the one step ahead
  ## prediction errors from 'bsts.object'.
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  style <- match.arg(style)
  if (missing(time)) {
    time <- index(bsts.object$original.series)
  }

  errors <- bsts.prediction.errors(bsts.object, burn = burn)
  if (style == "dynamic") {
    PlotDynamicDistribution(errors, time = time, ...)
  } else {
    TimeSeriesBoxplot(errors, time = time, ...)
  }
}

###----------------------------------------------------------------------
PlotBstsForecastDistribution <- function(bsts.object,
                                         burn = SuggestBurn(.1, bsts.object),
                                         time,
                                         style = c("dynamic", "boxplot"),
                                         show.actuals = TRUE,
                                         col.actuals = "blue",
                                         ...) {
  ## Plots the posterior distribution of the one-step-ahead forecasts
  ## for a bsts model.  This is the distribution of p(y[t+1] | y[1:t],
  ## \theta) averaged over p(\theta | y[1:T]).
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   show.actuals: If TRUE then the original values from the series
  ##     will be added to the plot.
  ##   col.actuals: The color to use when plotting original values
  ##     from the time series being modeled.
  ##   ...: Extra arguments passed to TimeSeriesBoxplot,
  ##     PlotDynamicDistribution, and points.
  ##
  ## Returns:
  ##   invisible NULL
  stopifnot(inherits(bsts.object, "bsts"))
  style = match.arg(style)
  if (missing(time)) {
    time = index(bsts.object$original.series)
  }

  errors <- bsts.prediction.errors(bsts.object, burn = burn)
  forecast <- t(as.numeric(bsts.object$original.series) - t(errors))
  if (style == "dynamic") {
    PlotDynamicDistribution(forecast, time = time, ...)
  } else {
    TimeSeriesBoxplot(forecast, time = time, ...)
  }

  if (show.actuals) {
    points(time, bsts.object$original.series, col = col.actuals, ...)
  }
  return(invisible(NULL))
}
###----------------------------------------------------------------------
PlotBstsResiduals <- function(bsts.object, burn = SuggestBurn(.1, bsts.object),
                              time, style = c("dynamic", "boxplot"),
                              ...) {
  ## Plots the posterior distribution of the residuals from the bsts
  ## model, after subtracting off the state effects (including
  ## regression effects).
  ## Args:
  ##   bsts.object:  An object of class 'bsts'.
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  stopifnot(inherits(bsts.object, "bsts"))
  style <- match.arg(style)

  if (missing(time)) {
    time <- index(bsts.object$original.series)
  }
  state <- bsts.object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
  residuals <- t(t(state) - as.numeric(bsts.object$original.series))
  if (style == "dynamic") {
    PlotDynamicDistribution(residuals, time = time, ...)
  } else {
    TimeSeriesBoxplot(residuals, time = time, ...)
  }
  return(invisible(NULL))
}

###----------------------------------------------------------------------
PlotDynamicRegression <- function(bsts.object,
                                  burn = SuggestBurn(.1, bsts.object),
                                  time = NULL,
                                  style = c("dynamic", "boxplot"),
                                  layout = c("square", "horizontal",
                                    "vertical"),
                                  ...) {
  ## Plot the coefficients of a dynamic regression state component.
  ## Args:
  ##   bsts.object: The bsts object containing the dynamic regression
  ##     state component to be plotted.
  ##
  ##   burn: The number of MCMC iterations to be discarded as burn-in.
  ##   time: An optional vector of values to plot on the time axis.
  ##   layout: A text string indicating whether the state components
  ##     plots should be laid out in a square (maximizing plot area),
  ##     vertically, or horizontally.
  ##   style: Either "dynamic", for dynamic distribution plots, or
  ##     "boxplot", for box plots.  Partial matching is allowed, so
  ##     "dyn" or "box" would work, for example.
  ##   ...: Additional arguments passed to PlotDynamicDistribution or
  ##     TimeSeriesBoxplot.
  stopifnot(inherits(bsts.object, "bsts"))
  if (!("dynamic.regression.coefficients" %in% names(bsts.object))) {
    stop("The model object does not contain a dynamic regression component.")
  }
  style <- match.arg(style)
  if (is.null(time)) {
    time <- index(bsts.object$original.series)
  }
  beta <- bsts.object$dynamic.regression.coefficients
  ndraws <- dim(beta)[1]
  number.of.variables <- dim(beta)[2]
  stopifnot(length(time) == dim(beta)[3])

  if (burn > 0) {
    beta <- beta[-(1:burn), , , drop = FALSE]
  }

  layout <- match.arg(layout)
  if (layout == "square") {
    num.rows <- floor(sqrt(number.of.variables))
    num.cols <- ceiling(number.of.variables / num.rows)
  } else if (layout == "vertical") {
    num.rows <- number.of.variables
    num.cols <- 1
  } else if (layout == "horizontal") {
    num.rows <- 1
    num.cols <- number.of.variables
  }
  original.par <- par(mfrow = c(num.rows, num.cols))
  on.exit(par(original.par))
  beta.names <- dimnames(beta)[[2]]

  for (variable in 1:number.of.variables) {
    if (style == "boxplot") {
      TimeSeriesBoxplot(beta[, variable, , ],
                        time = time,
                        ...)
    } else if (style == "dynamic") {
      PlotDynamicDistribution(beta[, variable, ],
                              time = time,
                              ...)
    }
    title(beta.names[variable])
  }
}

###----------------------------------------------------------------------
predict.bsts <- function(object, newdata, horizon = 1,
                         burn = SuggestBurn(.1, object),
                         na.action = na.exclude, olddata,
                         quantiles = c(.025, .975), ...) {
  ## Args:
  ##   object:  an object of class 'bsts' created using the function 'bsts'
  ##   newdata: a vector, matrix, or data frame containing the
  ##     predictor variables to use in making the prediction.  This is
  ##     only required if 'object' contains a regression compoent.  If
  ##     a data frame, it must include variables with the same names
  ##     as the data used to fit 'object'.  The first observation in
  ##     newdata is assumed to be one time unit after the end of the
  ##     last data used in fitting 'object', and the subsequent
  ##     observations are sequential time points.  If the regression
  ##     part of 'object' contains only a single predictor then
  ##     newdata can be a vector.  If 'newdata' is passed as a matrix
  ##     it is the caller's responsibility to ensure that it contains
  ##     the correct number of columns and that the columns correspond
  ##     to those in object$coefficients.
  ##   horizon: An integer specifying the number of periods into the
  ##     future you wish to predict.  If 'object' contains a regression
  ##     component then the forecast horizon is nrow(X) and this
  ##     argument is not used.
  ##   burn: An integer describing the number of MCMC iterations in
  ##     'object' to be discarded as burn-in.  If burn <= 0 then no
  ##     burn-in period will be discarded.
  ##   na.action: A function determining what should be done with
  ##     missing values in newdata.
  ##   olddata: An optional data frame including variables with the
  ##     same names as the data used to fit 'object'.  If 'olddata' is
  ##     missing then it is assumed that the first entry in 'newdata'
  ##     immediately follows the last entry in the training data for
  ##     'object'.  If 'olddata' is supplied then it will be filtered
  ##     to get the distribution of the next state before a prediction
  ##     is made, and it is assumed that the first entry in 'newdata'
  ##     comes immediately after the last entry in 'olddata'.
  ##   quantiles: A numeric vector of length 2 giving the lower and
  ##     upper quantiles to use for the forecast interval estimate.
  ##   ...: This is a dummy argument included to match the signature
  ##     of the generic predict() function.  It is not used.
  ## Returns:
  ##   An object of class 'bsts.prediction', which is a list with the
  ##   following elements:
  ##   mean: A numeric vector giving the posterior mean of the
  ##     predictive distribution at each time point.
  ##   interval: A two-column matrix giving the lower and upper limits
  ##     of the 95% prediction interval at each time point.
  ##   distribution: A matrix of draws from the posterior predictive
  ##     distribution.  Each column corresponds to a time point.  Each
  ##     row is an MCMC draw.
  ##   original.series: The original series used to fit 'object'.
  ##     This is used by the plot method to plot the original series
  ##     and the prediction together.

  stopifnot(inherits(object, "bsts"))

## TODO(stevescott): predict method for dynamic regressions

  if (object$has.regression) {
    if (missing(newdata)) {
      stop("You need to supply 'newdata' when making predictions with ",
           "a bsts object that has a regression component.")
    }
    if (is.data.frame(newdata)) {
      tt <- terms(object)
      Terms <- delete.response(tt)
      m <- model.frame(Terms, newdata, na.action = na.action,
                       xlev = object$xlevels)
      if (!is.null(cl <- attr(Terms, "dataClasses"))) .checkMFClasses(cl, m)
      X <- model.matrix(Terms, m, contrasts.arg = object$contrasts)

      if (nrow(X) != nrow(newdata)) {
        msg <- paste("Some entries in newdata have missing values, and  will",
                     "be omitted from the prediction.")
        warning(msg)
      }
      if (ncol(X) != ncol(object$coefficients)) {
        stop("Wrong number of columns in newdata.  ",
             "(Check that variable names match?)")
      }

    } else {
      X <- as.matrix(newdata)
      if (ncol(X) == ncol(object$coefficients) - 1) {
        X <- cbind(1, X)
      }
      if (ncol(X) != ncol(object$coefficients)) {
        stop("Wrong number of columns in newdata")
      }

      na.rows <- rowSums(is.na(X)) > 0
      if (any(na.rows)) {
        warning("Entries in newdata containing missing values will be",
                "omitted from the prediction")
        X <- X[!na.rows, ]
      }
    }


    ## still need to do the predictions
    if (!missing(olddata)) {
      ## Need to go through the formula business to get y and x for olddata
      oldm <- model.frame(tt, olddata, na.action = na.action,
                          xlev = object$xlevels)
      if (!is.null(cl <- attr(tt, "dataClasses"))) .checkMFClasses(cl, oldm)

      ## oldX and oldy should be the model matrix and response for the
      ## old data
      oldX <- model.matrix(Terms, oldm, contrasts.arg = object$contrasts)
      oldy <- model.response(oldm, "numeric")
    } else {
      oldX <- NULL
      oldy <- NULL
    }

    predictive.distribution <-
      .Call("bsts_predict_state_space_regression_",
            object,
            X,
            burn,
            oldX,
            oldy,
            PACKAGE = "bsts")
  } else {
    ## Handle the no-regression case.
    niter <- length(object$sigma.obs)
    stopifnot(niter > burn)
    if (missing(olddata)) {
      olddata <- NULL
    }
    stopifnot(is.null(olddata) || is.numeric(olddata))

    predictive.distribution <-
      .Call("bsts_predict_state_space_model_",
            object,
            horizon,
            burn,
            olddata,
            PACKAGE = "bsts")
  }

  ans <- list("mean" = colMeans(predictive.distribution),
              "median" = apply(predictive.distribution, 2, median),
              "interval" = apply(predictive.distribution, 2,
                                 quantile, quantiles),
              "distribution" = predictive.distribution,
              "original.series" = object$original.series)
  class(ans) <- "bsts.prediction"
  return(ans)
}

###----------------------------------------------------------------------
bsts.prediction.errors <- function(bsts.object,
                                   newdata,
                                   burn = SuggestBurn(.1, bsts.object),
                                   na.action = na.omit) {
  ## Returns the posterior distribution of the one-step-ahead
  ## prediction errors from the bsts.object.  The errors are organized
  ## as a matrix, with rows corresponding to MCMC iteration, and
  ## columns corresponding to time.
  ## Args:
  ##   bsts.object:  An object created by a call to 'bsts'
  ##   newdata: An optional data.frame containing data that is assumed
  ##     to immediatly follow the training data (in time).  If
  ##     'newdata' is supplied the one-step-ahead prediction errors
  ##     will be relative to the responses in 'newdata'.  Otherwise
  ##     they will be relative to the training data.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   na.action: A function describing what should be done with NA
  ##     elements of newdata.
  ## Returns:
  ##   A matrix of prediction errors, with rows corresponding to MCMC
  ##   iteration, and columns to time.
  if (!missing(newdata)) {
    return(bsts.holdout.prediction.errors(bsts.object,
                                          newdata,
                                          burn,
                                          na.action))
  }

  if (!is.null(bsts.object$one.step.prediction.errors)) {
    return(bsts.object$one.step.prediction.errors)
  }

  if (bsts.object$has.regression) {
    errors <- .Call("bsts_one_step_training_prediction_errors_",
                    bsts.object, burn,
                    PACKAGE = "bsts")
  } else {
    errors <- .Call("bsts_one_step_prediction_errors_no_regression_",
                    bsts.object, NULL, burn,
                    PACKAGE = "bsts")
  }
  return(errors)
}

###----------------------------------------------------------------------
bsts.holdout.prediction.errors <- function(bsts.object,
                                           newdata,
                                           burn = SuggestBurn(.1, bsts.object),
                                           na.action = na.omit) {
  ## Return the one step ahead prediction errors for the holdout
  ## sample in 'newdata' which is assumed to follow immediately
  ## after the training data used to fit 'bsts.object'.
  ## Args:
  ##   bsts.object: An object of class 'bsts' for which prediction
  ##     errors are desired.
  ##   newdata: A holdout sample of data to be predicted.
  ##     If 'bsts.object' has a regression component then 'newdata'
  ##     must be a data.frame containing all the variables that
  ##     appear in the model formula found in 'bsts.object'.
  ##     Otherwise, 'newdata' is a numeric vector.
  ##   burn: The number of MCMC iterations to be discarded as
  ##     burn-in.  If burn == 0 then no burn-in sample will be
  ##     discarded.
  ##   na.action: A function indicating what should be done with
  ##     NA's in 'newdata'.
  ## Returns:
  ##   A matrix of draws of the one-step ahead prediction errors for
  ##   the holdout sample.  Rows correspond to MCMC iteration.
  ##   Columns to time.
  stopifnot(inherits(bsts.object, "bsts"))
  stopifnot(burn >= 0)
  if (bsts.object$has.regression) {

    Terms <- terms(bsts.object)
    m <- model.frame(Terms,
                     newdata,
                     na.action = na.action,
                     xlev = bsts.object$xlevels)
    if (!is.null(cl <- attr(Terms, "dataClasses"))) .checkMFClasses(cl, m)
    X <- model.matrix(Terms, m, contrasts.arg = bsts.object$contrasts)
    y <- model.response(m, "numeric")

    if (nrow(X) != nrow(newdata)) {
      warning("Some entries in newdata have missing values, and  will ",
              "be omitted from the prediction.")
    }
    stopifnot(length(y) == nrow(X))

    ans <- .Call("bsts_one_step_holdout_prediction_errors_",
                 bsts.object,
                 X,
                 y,
                 burn,
                 PACKAGE = "bsts")
  } else {
    ## Handle no regression case.
    stopifnot(is.numeric(newdata))
    ans <- .Call("bsts_one_step_prediction_errors_no_regression_",
                 bsts.object,
                 newdata,
                 burn,
                 PACKAGE = "bsts")
  }
  class(ans) <- "bsts.prediction.errors"
  return(ans)
}

###----------------------------------------------------------------------
plot.bsts.prediction <- function(x,
                                 y = NULL,
                                 burn = 0,
                                 plot.original = TRUE,
                                 median.color = "blue",
                                 median.type = 1,
                                 median.width = 3,
                                 interval.quantiles = c(.025, .975),
                                 interval.color = "green",
                                 interval.type = 2,
                                 interval.width = 2,
                                 style = c("dynamic", "boxplot"),
                                 ylim = NULL,
                                 ...) {
  ## Plots the posterior predictive distribution found in the
  ## 'prediction' object.
  ## Args:
  ##   x: An object with class 'bsts.prediction', generated
  ##     using the 'predict' method for a 'bsts' object.
  ##   y: A dummy argument needed to match the signature of the plot()
  ##     generic function.  It is not used.
  ##   burn: The number of observations you wish to discard as burn-in
  ##     from the posterior predictive distribution.  This is in
  ##     addition to the burn-in discarded using predict.bsts.
  ##   plot.original: Logical.  If true then the prediction is plotted
  ##     after a time series plot of the original series.  Otherwise,
  ##     the prediction fills the entire plot.
  ##   median.color: The color to use for the posterior median of the
  ##     prediction.
  ##   median.type: The type of line (lty) to use for the posterior median
  ##     of the prediction.
  ##   median.width: The width of line (lwd) to use for the posterior median
  ##     of the prediction.
  ##   interval.quantiles: The lower and upper limits of the credible
  ##     interval to be plotted.
  ##   interval.color: The color to use for the upper and lower limits
  ##     of the 95% credible interval for the prediction.
  ##   interval.type: The type of line (lty) to use for the upper and
  ##     lower limits of the 95% credible inerval for of the
  ##     prediction.
  ##   interval.width: The width of line (lwd) to use for the upper and
  ##     lower limits of the 95% credible inerval for of the
  ##     prediction.
  ##   style: What type of plot should be produced?  A
  ##     DynamicDistribution plot, or a time series boxplot.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments to be passed to PlotDynamicDistribution()
  ##     and lines().
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.

  prediction <- x
  if (burn > 0) {
    prediction$distribution <-
      prediction$distribution[-(1:burn), , drop = FALSE]
    prediction$median <- apply(prediction$distribution, 2, median)
    prediction$interval <- apply(prediction$distribution, 2,
                                 quantile, c(.025, .975))
  }
  prediction$interval <- apply(prediction$distribution, 2,
                               quantile, interval.quantiles)

  n0 <- length(prediction$original.series)
  n1 <- ncol(prediction$distribution)

  time <- index(prediction$original.series)
  deltat <- tail(diff(tail(time, 2)), 1)

  if (is.null(ylim)) {
    ylim <- range(prediction$distribution,
                  prediction$original.series)
  }

  if (plot.original) {
    pred.time <- tail(time, 1) + (1:n1) * deltat
    plot(time,
         prediction$original.series,
         type = "l",
         xlim = range(time, pred.time),
         ylim = ylim)
  } else {
    pred.time <- tail(time, 1) + (1:n1) * deltat
  }

  style <- match.arg(style)
  if (style == "dynamic") {
    PlotDynamicDistribution(curves = prediction$distribution,
                            time = pred.time,
                            add = plot.original,
                            ylim = ylim,
                            ...)
  } else {
    TimeSeriesBoxplot(prediction$distribution,
                      time = pred.time,
                      add = plot.original,
                      ylim = ylim,
                      ...)
  }
  lines(pred.time, prediction$median, col = median.color,
        lty = median.type, lwd = median.width, ...)
  for (i in 1:nrow(prediction$interval)) {
    lines(pred.time, prediction$interval[i, ], col = interval.color,
          lty = interval.type, lwd = interval.width, ...)
  }
  return(invisible(NULL))
}

StateSizes <- function(state.specification) {
  ## Returns a vector giving the number of dimensions used by each state
  ## component in the state vector.
  ## Args:
  ##   state.specification: a vector of state specification elements.
  ##     This most likely comes from the state.specification element
  ##     of a bsts.object
  ## Returns:
  ##   A numeric vector giving the dimension of each state component.
  state.component.names <- sapply(state.specification, function(x) x$name)
  state.sizes <- sapply(state.specification, function(x) x$size)
  names(state.sizes) <- state.component.names
  return(state.sizes)
}

SuggestBurn <- function(proportion, bsts.object) {
  ## Suggests a size of a burn-in sample to be discarded from the MCMC
  ## run.
  ## Args:
  ##   proportion: A number between 0 and 1.  The fraction of the run
  ##     to discard.
  ##   bsts.object:  An object of class 'bsts'.
  ## Returns:
  ##   The number of iterations to discard.
  return(floor(proportion * length(bsts.object$sigma.obs)))
}

Shorten <- function(words) {
  ## Removes a prefix and suffix common to all elements of 'words'.
  ## Args:
  ##   words:  A character vector.
  ##
  ## Returns:
  ##   'words' with common prefixes and suffixes removed.
  ##
  ## Details:
  ##   Then intent is to use this function on files from the same
  ##   directory with similar suffixes.
  ##
  ##   Shorten(c("/usr/common/foo.tex", "/usr/common/barbarian.tex")
  ##   produces c("foo", "barbarian")
  if (is.null(words)) return (NULL)
  stopifnot(is.character(words))
  if (length(unique(words)) == 1) {
    ## If all the words are the same don't do any shortening.
    return(words)
  }

  first.letters <- substring(words, 1, 1)
  while (all(first.letters == first.letters[1])) {
    words <- substring(words, 2)
    first.letters <- substring(words, 1, 1)
  }

  word.length <- nchar(words)
  last.letters <- substring(words, word.length, word.length)
  while (all(last.letters == last.letters[1])) {
    words <- substring(words, 1, word.length - 1)
    word.length <- word.length - 1
    last.letters <- substring(words, word.length, word.length)
  }

  return(words)
}
