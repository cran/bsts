// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include "utils.h"
#include "r_interface/check_interrupt.h"
#include "r_interface/error.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/seed_rng_from_R.hpp"
#include "r_interface/handle_exception.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceRegressionSampler.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"

#include "R_ext/Arith.h"  // for ISNA

using BOOM::BregVsSampler;
using BOOM::DoubleData;
using BOOM::Mat;
using BOOM::Ptr;
using BOOM::RegressionData;
using BOOM::RegressionModel;
using BOOM::Spd;
using BOOM::StateModel;
using BOOM::StateSpaceModel;
using BOOM::StateSpaceModelBase;
using BOOM::StateSpacePosteriorSampler;
using BOOM::StateSpaceRegressionModel;
using BOOM::SubMatrix;
using BOOM::Vec;
using BOOM::VectorParams;
using BOOM::ZeroMeanGaussianConjSampler;
using BOOM::ZeroMeanGaussianModel;

using BOOM::ToBoomVector;
using BOOM::ToVectorBool;
using BOOM::getListElement;

using BOOM::NativeMatrixListElement;
using BOOM::NativeVectorListElement;
using BOOM::RListIoElement;
using BOOM::RListIoManager;
using BOOM::StandardDeviationListElement;
using BOOM::VectorListElement;

using BOOM::RCheckInterrupt;
using BOOM::RErrorReporter;

namespace bsts {
RListIoManager SpecifyStateSpaceModel(
    Ptr<StateSpaceModel>,
    SEXP state_specification,
    SEXP r_sigma_prior,
    Vec *final_state,
    bool save_state_history,
    bool save_prediction_errors);

// A callback to manage recording the contributions from each state
// component.
class StateContributionCallback : public BOOM::MatrixIoCallback {
 public:
  explicit StateContributionCallback(BOOM::StateSpaceModelBase *model)
      : model_(model) {}
  virtual int nrow()const {return model_->nstate();}
  virtual int ncol()const {return model_->time_dimension();}
  virtual BOOM::Mat get_matrix()const {
    BOOM::Mat ans(nrow(), ncol());
    for (int state = 0; state < model_->nstate(); ++state) {
      ans.row(state) = model_->state_contribution(state);
    }
    return ans;
  }
 private:
  BOOM::StateSpaceModelBase *model_;
};

//======================================================================
// Add the state models and specify the prior for the variance of
// the observation model.
// Args:
//   model:  The StateSpaceModel to be specified
//   state_specification: An R list specifying the components of state
//     to be added.  See the comments in create_state_model.cc.
//   rsigma_prior: This can be NULL if the model is being created for
//     purposes other than posterior sampling.  Otherwise it must be
//     an R object of class SdPrior giving the prior distribution for
//     the residual standard deviation.
//   final_state: A pointer to a BOOM::Vec that will be used to hold
//     the sampled values of the final state when reading through an
//     MCMC stream.  If the model is being used for output only then
//     this can be NULL.
//   save_state_history: If true then the io_manager will record the
//     contribution of each state model towards the mean of each
//     observation.  Setting this parameter to 'false' will reduce the
//     size of the R object, but prohibit several useful options in
//     the R function plot.bsts.
// Returns:
//   The return value is the io_manager responsible for managing the
//   stream of MCMC samples.
RListIoManager SpecifyStateSpaceModel(Ptr<StateSpaceModel> model,
                                      SEXP state_specification,
                                      SEXP rsigma_prior,
                                      BOOM::Vec *final_state,
                                      bool save_state_history,
                                      bool save_prediction_errors) {
  RListIoManager io_manager;
  io_manager.add_list_element(new StandardDeviationListElement(
      model->observation_model()->Sigsq_prm(),
      "sigma.obs"));

  BOOM::RInterface::StateModelFactory factory(&io_manager, model.get());
  factory.AddState(state_specification);

  ZeroMeanGaussianModel *obs = model->observation_model();
  if (!Rf_isNull(rsigma_prior)) {
    BOOM::RInterface::SdPrior sigma_prior(rsigma_prior);
    BOOM::Ptr<ZeroMeanGaussianConjSampler> sigma_sampler(
        new ZeroMeanGaussianConjSampler(
            obs,
            sigma_prior.prior_df(),
            sigma_prior.prior_guess()));
    sigma_sampler->set_sigma_upper_limit(sigma_prior.upper_limit());
    obs->set_method(sigma_sampler);
    model->set_method(new StateSpacePosteriorSampler(model.get()));
  }

  factory.SaveFinalState(final_state);

  if (save_state_history) {
    io_manager.add_list_element(
        new NativeMatrixListElement(
            new StateContributionCallback(model.get()),
            "state.contributions",
            NULL));
  }

  if (save_prediction_errors) {
    // The final NULL argument is because we will not be streaming
    // prediction errors in future calculations.  They are for
    // reporting only.  As usual, the rows of the matrix correspond to
    // MCMC iterations, so the columns represent time.
    io_manager.add_list_element(
        new BOOM::NativeVectorListElement(
            // TODO(stevescott): Write PredictionErrorCallback.
            new PredictionErrorCallback(model.get()),
            "one.step.prediction.errors",
            NULL));
  }
  return io_manager;
}

namespace boomr = BOOM::RInterface;

using BOOM::RInterface::handle_exception;
using BOOM::RInterface::handle_unknown_exception;
extern "C" {
  //======================================================================
  // Primary C++ interface for StateSpaceRegression.  Note the
  // difference between this and fit_state_space_regression_.  This
  // one has no predictor variables other than the time series being
  // modeled.
  // Args:
  //   ry: A numeric vector.  The time series to be modeled.
  //   ry_is_observed: An R logical vector indicating which elements
  //     of y are non-NA.  NA elements of y correspond to missing
  //     observations that should be smoothed over by the Kalman
  //     filter.
  //   rstate_specification: A list of state specifications created by
  //     AddLocalLinearTrend, AddSeasonal, etc.  See the help file
  //     StateSpecification.Rd.
  //   rsave_state_contribution_flag: A logical scalar indicating
  //     whether the contributions from each state space model should
  //     be saved.  Setting this to FALSE will reduce the size of the
  //     returned object, but prohibit several useful options in the R
  //     function plot.bsts.
  //   rniter: An integer scalar giving the desired number of MCMC
  //     iterations.
  //   rsigma_prior: An object of class SdPrior specifying the prior
  //     distribution on the residual standard deviation.
  //   rping: An integer scalar.  If rping > 0 then a status update
  //     will be printed to the screen every rping MCMC iterations.
  //   rseed: An integer to use as the C++ random seed, or NULL.  If
  //     NULL then the C++ seed will be set using the clock.
  // Returns:
  //   An R object containing the posterior draws defining the model.
  SEXP bsts_fit_state_space_model_(
      SEXP ry,
      SEXP ry_is_observed,
      SEXP rstate_specification,
      SEXP rsave_state_contribution_flag,
      SEXP rsave_prediction_errors_flag,
      SEXP rniter,
      SEXP rsigma_prior,
      SEXP rping,
      SEXP rseed) {
    RErrorReporter error_reporter;
    try {
      BOOM::RInterface::seed_rng_from_R(rseed);
      BOOM::Vec y(ToBoomVector(ry));
      std::vector<bool> y_is_observed = ToVectorBool(ry_is_observed);
      int niter = lround(Rf_asReal(rniter));
      int ping = lround(Rf_asReal(rping));

      BOOM::Ptr<BOOM::StateSpaceModel> model(
          new BOOM::StateSpaceModel(y, y_is_observed));

      Vec final_state;
      RListIoManager io_manager = SpecifyStateSpaceModel(
          model,
          rstate_specification,
          rsigma_prior,
          &final_state,
          Rf_asLogical(rsave_state_contribution_flag),
          Rf_asLogical(rsave_prediction_errors_flag));

      // Do one posterior sampling step before getting ready to write.
      // This will ensure that any dynamically allocated objects have
      // the correct size before any R memory gets allocated in the
      // call to prepare_to_write().
      model->sample_posterior();

      SEXP ans;
      PROTECT(ans = io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        BOOM::print_R_timestamp(i, ping);
        try {
          model->sample_posterior();
          io_manager.write();
        } catch(std::exception &e) {
          std::ostringstream err;
          err << "state_space.cc caught an exception with the following "
              << "error message in MCMC "
              << "iteration " << i << ".  Aborting." << std::endl
              << e.what() << std::endl;
          error_reporter.SetError(err.str());
          UNPROTECT(1);
          return ans;
        }
      }
      UNPROTECT(1);
      return ans;
    } catch(std::exception &e) {
      handle_exception(e);
    } catch(...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

  //======================================================================
  // Prediction for state space models.
  // Args:
  //   bsts_object:  An R object returned by the 'bsts' function.
  //   rhorizon: An integer scalar.  The forecast horizon.  I.e. the
  //     number of time periods ahead to forecast.
  //   rburn: An integer scalar.  The number of MCMC iterations in
  //     'bsts_object' to discard as burn-in.
  //   robserved_data: If this is a numeric vector then it is assumed
  //     to be the vector of observed data to be used as the basis for
  //     the forecast.  If it is NULL then the forecast is based on the
  //     training data used to fit 'bsts_object'.
  // Returns:
  //   A matrix with niter - 'rburn' rows (where 'niter' is the number
  //   of MCMC iterations in 'bsts_object'), and 'horizon' columns.  Each
  //   row is a draw from the posterior predictive forecast
  //   distribution, and each column is a time point in that
  //   distribution.
  SEXP bsts_predict_state_space_model_(
      SEXP bsts_object,
      SEXP rhorizon,
      SEXP rburn,
      SEXP robserved_data) {
    try {
      Ptr<StateSpaceModel> model(new StateSpaceModel);
      Vec final_state;  // to be streamed from info table
      SEXP state_specification = getListElement(
          bsts_object, "state.specification");

      // When specifying the state space model, some values are needed
      // from the prior distribution.  However because we will not be
      // doing any posterior sampling, we can just specify some dummy
      // values from the prior.
      RListIoManager io_manager = SpecifyStateSpaceModel(
          model,
          state_specification,
          R_NilValue,
          &final_state,
          false,
          false);
      io_manager.prepare_to_stream(bsts_object);
      int burn = lround(Rf_asReal(rburn));
      if (burn > 0) {
        io_manager.advance(burn);
      }
      if (burn < 0) {
        burn = 0;
      }

      bool have_observed_data = !Rf_isNull(robserved_data);
      Vec observed_data;
      if (have_observed_data) {
        observed_data = ToBoomVector(robserved_data);
        for (int i = 0; i < observed_data.size(); ++i) {
          Ptr<DoubleData> dp(new DoubleData(observed_data[i]));
          model->add_data(dp);
        }
      }

      int niter = LENGTH(getListElement(bsts_object, "sigma.obs"));
      int horizon = lround(Rf_asReal(rhorizon));
      SEXP forecast;
      PROTECT(forecast = Rf_allocMatrix(REALSXP, niter - burn, horizon));
      SubMatrix forecast_view(REAL(forecast), niter - burn, horizon);
      for (int i = burn; i < niter; ++i) {
        io_manager.stream();
        if (have_observed_data) {
          forecast_view.row(i-burn) =
              model->simulate_forecast_given_observed_data(
                  horizon, observed_data);
        } else {
          forecast_view.row(i-burn) = model->simulate_forecast(
              horizon, final_state);
        }
      }
      UNPROTECT(1);
      return forecast;
    } catch(std::exception &e) {
      handle_exception(e);
    } catch(...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

  //======================================================================
  // Compute the one-step-ahead prediction errors from either the
  // training data or a holdout sample.
  // Args:
  //   object: An R object computed from 'bsts'.  It must not contain
  //     a regression component.
  //   rnewY: Either the NULL object, in which case the prediction
  //     errors from the training data are returned, or a numeric
  //     vector containing a holdout sample immediately following the
  //     training data.
  //   rburn: An integer (passed from R) indicating the number of MCMC
  //     iterations to discard as burn-in.  If rniter <= 0 then no
  //     burn-in set is discarded.
  // Returns:
  //   An R matrix containing draws of the one-step-ahead prediction
  //   errors.  Rows correspond to MCMC iteration.  Columns correspond
  //   to time.  The errors are with respect to the training data if
  //   'rnewY' is NULL, otherwise they are with respect to rnewY.

  SEXP bsts_one_step_prediction_errors_no_regression_(
      SEXP object,
      SEXP rnewY,
      SEXP rburn) {
    try {
    Vec y = ToBoomVector(getListElement(object, "original.series"));
    Ptr<StateSpaceModel> model(new StateSpaceModel(y));
    SEXP state_specification = getListElement(object, "state.specification");
    Vec final_state;
    RListIoManager io_manager = SpecifyStateSpaceModel(
        model,
        state_specification,
        R_NilValue,
        &final_state,
        false,
        false);

    int burn = Rf_asInteger(rburn);
    io_manager.prepare_to_stream(object);
    if (burn > 0) io_manager.advance(burn);
    if (burn < 0) burn = 0;
    int niter = LENGTH(getListElement(object, "sigma.obs"));
    int time_horizon;
    Vec newY;
    bool holdout_sample;
    if (Rf_isNull(rnewY)) {
      holdout_sample = false;
      time_horizon = y.size();
    } else {
      holdout_sample = true;
      newY = ToBoomVector(rnewY);
      time_horizon = newY.size();
    }

    SEXP errors;
    PROTECT(errors = Rf_allocMatrix(REALSXP, niter - burn, time_horizon));
    SubMatrix error_view(REAL(errors), niter - burn, time_horizon);
    for (int i = burn; i < niter; ++i) {
      io_manager.stream();
      if (holdout_sample) {
        error_view.row(i - burn) = model->one_step_holdout_prediction_errors(
            newY, final_state);
      } else {
        error_view.row(i - burn) = model->one_step_prediction_errors();
      }
    }
    UNPROTECT(1);
    return errors;
    } catch(std::exception &e) {
      handle_exception(e);
    } catch(...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }
}

}  // namespace bsts
