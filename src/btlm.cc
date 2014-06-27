// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include "utils.h"

#include "r_interface/check_interrupt.h"
#include "r_interface/error.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "cpputil/report_error.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceRegressionSampler.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"

#include "R_ext/Arith.h"  // for ISNA

using BOOM::BregVsSampler;
using BOOM::Mat;
using BOOM::Ptr;
using BOOM::Spd;
using BOOM::SubMatrix;
using BOOM::Vec;

using BOOM::RegressionData;
using BOOM::RegressionModel;
using BOOM::StateSpaceModel;
using BOOM::StateSpaceModelBase;
using BOOM::StateSpacePosteriorSampler;
using BOOM::StateSpaceRegressionModel;

using BOOM::GetStringFromList;
using BOOM::ToBoomMatrix;
using BOOM::ToBoomSpd;
using BOOM::ToBoomVector;
using BOOM::ToVectorBool;
using BOOM::getListElement;

using BOOM::NativeMatrixListElement;
using BOOM::NativeVectorListElement;
using BOOM::RListIoManager;
using BOOM::StandardDeviationListElement;
using BOOM::VectorListElement;
using BOOM::GlmCoefsListElement;

using BOOM::RCheckInterrupt;
using BOOM::RErrorReporter;

using BOOM::RInterface::handle_exception;
using BOOM::RInterface::handle_unknown_exception;

namespace bsts {
void AddRegressionPriorAndSetSampler(Ptr<StateSpaceRegressionModel>, SEXP);
void AddRegressionData(Ptr<StateSpaceRegressionModel>, SEXP bsts_object);

RListIoManager SpecifyStateSpaceRegressionModel(
    Ptr<StateSpaceRegressionModel>,
    SEXP state_specification,
    SEXP regression_prior,
    Vec *final_state,
    bool save_state_history,
    bool save_prediction_errors);

// A callback class to be used for saving contributions from the
// individual state models.
class RegressionStateContributionCallback : public BOOM::MatrixIoCallback {
 public:
  explicit RegressionStateContributionCallback(
      BOOM::StateSpaceRegressionModel *model)
      : model_(model) {}

  // There is one row for each stored state model, plus an additional
  // row for the regression effects.  The fact that the regression is
  // implemented differently than the other elements of state is an
  // unfortunate implementation detail.
  virtual int nrow()const {return 1 + model_->nstate();}
  virtual int ncol()const {return model_->time_dimension();}
  virtual BOOM::Mat get_matrix()const {
    BOOM::Mat ans(nrow(), ncol());
    for (int state = 0; state < model_->nstate(); ++state) {
      ans.row(state) = model_->state_contribution(state);
    }
    ans.last_row() = model_->regression_contribution();
    return ans;
  }
 private:
  BOOM::StateSpaceRegressionModel * model_;
};

//======================================================================
// Completes the specification of the given
// StateSpaceRegressionModel, by adding in the state models and
// regression prior created by R.
// Args:
//   model:  The StateSpaceRegressionModel to be specified.
//   state_specification: An R list specifying the components of state
//     to be added.  See the comments in create_state_model.cc.
//   regression_prior: An R list.  See comments in
//     AddRegressionPriorAndSetSampler, below.
//   final_state: A pointer to a BOOM::Vec that can be used to read in
//     the value of the state for the final time point.  This is only
//     needed if the model is going to be reading previously sampled
//     MCMC output, otherwise it can be NULL.
//   save_state_history: If true then MCMC simulations will record the
//     value of the state vector at the time of the final observation
//     (which is useful for forecasting later).
// Returns:
//   An RListIoManager containing information needed to allocate space,
//     record, and stream the model parameters and related information.
RListIoManager SpecifyStateSpaceRegressionModel(
    Ptr<StateSpaceRegressionModel> model,
    SEXP state_specification,
    SEXP regression_prior,
    Vec *final_state,
    bool save_state_history,
    bool save_prediction_errors) {
  RListIoManager io_manager;
  Ptr<RegressionModel> regression(model->regression_model());

  io_manager.add_list_element(
      new GlmCoefsListElement(regression->coef_prm(), "coefficients"));
  io_manager.add_list_element(
      new StandardDeviationListElement(regression->Sigsq_prm(),
                                       "sigma.obs"));

  BOOM::RInterface::StateModelFactory factory(&io_manager, model.get());
  factory.AddState(state_specification);
  AddRegressionPriorAndSetSampler(model, regression_prior);
  model->set_method(new StateSpacePosteriorSampler(model.get()));

  factory.SaveFinalState(final_state);

  if (save_state_history) {
    // The final NULL argument is because we won't be streaming state
    // contributions in future calculations.  They are for reporting
    // only.
    io_manager.add_list_element(
        new BOOM::NativeMatrixListElement(
            new RegressionStateContributionCallback(
                model.get()),
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
            new PredictionErrorCallback(model.get()),
            "one.step.prediction.errors",
            NULL));
  }
  return io_manager;
}

//======================================================================
// Adds a regression prior to the RegressionModel managed by model.
// Args:
//   model: The StateSpaceRegressionModel that needs a regression
//     prior assigned to it.
//   regression_prior: An R object created by SpikeSlabPrior,
//     which is part of the BoomSpikeSlab package.
void AddRegressionPriorAndSetSampler(Ptr<StateSpaceRegressionModel> model,
                                     SEXP regression_prior) {
  Vec prior_inclusion_probs(ToBoomVector(getListElement(
      regression_prior, "prior.inclusion.probabilities")));
  Vec mu(ToBoomVector(getListElement(
      regression_prior, "mu")));
  Spd Siginv(ToBoomSpd(getListElement(
      regression_prior, "siginv")));
  double prior_df = Rf_asReal(getListElement(regression_prior, "prior.df"));
  double prior_sigma_guess = Rf_asReal(getListElement(
      regression_prior, "sigma.guess"));
  int max_flips = Rf_asInteger(getListElement(regression_prior, "max.flips"));

  Ptr<RegressionModel> regression(model->regression_model());
  // Initialize the model to be empty, except for variables that are
  // known to be present with probability 1.
  regression->coef().drop_all();
  for (int i = 0; i < prior_inclusion_probs.size(); ++i) {
    if (prior_inclusion_probs[i] >= 1.0) {
      regression->coef().add(i);
    }
  }

  Ptr<BregVsSampler> sampler(new BregVsSampler(
      regression.get(),
      mu,
      Siginv,
      prior_sigma_guess,
      prior_df,
      prior_inclusion_probs));
  sampler->limit_model_selection(max_flips);
  regression->set_method(sampler);
}

//----------------------------------------------------------------------
// Helper function that populates a StateSpaceRegressionModel with
// data, based on the data stored in a 'bsts' object.
// Args:
//   model:  The state space model to be populated with data.
//   object:  The bsts object from which to extract the data.
void AddRegressionData(Ptr<StateSpaceRegressionModel> model,
                       SEXP object) {
  // Need to add dummy values for old data so the model can get
  // the 'time' subscript right for the prediction.
  Mat X(ToBoomMatrix(getListElement(object, "design.matrix")));
  Vec y(ToBoomVector(getListElement(object, "original.series")));
  for (int i = 0; i < y.size(); ++i) {
    model->add_data(Ptr<RegressionData>(new RegressionData(y[i], X.row(i))));
  }
}

extern "C" {
  //======================================================================
  // Primary C++ interface to the R bsts function when using a formula.
  // Args:
  //   rx: An R matrix created using model.matrix.  This is the design
  //     matrix for the regression.
  //   ry: An R vector of responses created using model.response from
  //     the R function bsts().
  //   ry_is_observed: An R logical vector indicating which elements
  //     of y are non-NA.  NA elements of y correspond to missing
  //     observations that should be smoothed over by the Kalman
  //     filter.
  //   state_specification: An R list of state model specifications.  See
  //     the R documentation and the comments in create_state_model.cc
  //   save_state_contribution_flag: An R logical indicating whether
  //     contributions from the individual state model components
  //     should be saved.  Saving them requires storage of order
  //     (number of time points) * (number of state models) * (number
  //     of mcmc iterations)
  //   regression_prior: An R object specifying the prior for the
  //     regression coefficients and observation variance.  This is
  //     assumed to have been created by SpikeSlabPrior, part of
  //     the BoomSpikeSlab package.
  //   rniter:  An R scalar giving the desired number of MCMC iterations.
  //   rping: An R integer indicating the frequency with which
  //     progress reports get printed.  E.g. setting rping = 100 will
  //     print a status message with a time and iteration stamp every
  //     100 iterations.  If you don't want these messages set rping < 0.
  //   rseed: An integer to use as the C++ random seed, or NULL.  If
  //     NULL then the C++ seed will be set using the clock.
  // Returns:
  //   An R object containing the posterior draws defining the model.
  SEXP bsts_fit_state_space_regression_(
      SEXP rx,
      SEXP ry,
      SEXP ry_is_observed,
      SEXP rstate_specification,
      SEXP rsave_state_contribution_flag,
      SEXP rsave_prediction_errors_flag,
      SEXP rregression_prior,
      SEXP rniter,
      SEXP rping,
      SEXP rseed) {
    RErrorReporter error_reporter;
    try {
      // The try block is needed to catch any errors that come from
      // boom_r_tools.  E.g. ToBoomVector...
      BOOM::RInterface::seed_rng_from_R(rseed);
      BOOM::Mat X = ToBoomMatrix(rx);
      BOOM::Vec y = ToBoomVector(ry);

      std::vector<bool> y_is_observed = ToVectorBool(ry_is_observed);

      if (nrow(X) != y.size()) {
        error_reporter.SetError("error in bsts::fit_state_space_regression_:  "
                                "number of rows in X does not match length of "
                                "y");
        return R_NilValue;
      }

      Ptr<StateSpaceRegressionModel> model(
          new StateSpaceRegressionModel(y, X, y_is_observed));

      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              rstate_specification,
              rregression_prior,
              NULL,
              Rf_asLogical(rsave_state_contribution_flag),
              Rf_asLogical(rsave_prediction_errors_flag));

      // Do one posterior sampling step before getting ready to write.
      // This will ensure that any dynamically allocated objects have
      // the correct size before any R memory gets allocated in the
      // call to prepare_to_write().
      model->sample_posterior();

      int niter = lround(Rf_asReal(rniter));
      int ping = lround(Rf_asReal(rping));
      SEXP ans = PROTECT(io_manager.prepare_to_write(niter));
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
          err << "btlm.cc caught an exception with the following "
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
  // The primary C++ interface to the R function predict.bsts when
  // using a formula.
  // Args:
  //   object: The R object returned by the R function bsts.  This is
  //     the object returned by fit_state_space_regression_, with
  //     other stuff added on by bsts() on the R side.
  //   rnewX: An R matrix giving the predictor variables to be used
  //     in the forecast.
  //   rburn: An R scalar giving the number of MCMC iterations to be
  //     discarded before starting the forecast.
  //   roldX, roldy: The default is to start the predictions from the
  //     next observation following the training data.  In that case
  //     roldX and roldy should be NULL.  If predictions should be
  //     passed based on a different set of training data, then pass
  //     those data in as roldX and roldY.
  // TODO(stevescott): This is confusing.  Understand it better, and
  //     then explain it better.
  // Returns:
  // An R matrix with rows corresponding to MCMC iterations, and
  // columns corresponding to forecast time.
  SEXP bsts_predict_state_space_regression_(
      SEXP object,
      SEXP rnewX,
      SEXP rburn,
      SEXP roldX,
      SEXP roldy) {
    SEXP forecast;
    try {
      BOOM::Mat newX(ToBoomMatrix(rnewX));
      int xdim = ncol(newX);
      Ptr<StateSpaceRegressionModel> model(
          new StateSpaceRegressionModel(xdim));

      Vec final_state;
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              &final_state,
              false,       // Don't save state contributions.
              false);      // Don't save prediction errors.

      bool have_old_data(false);
      if (!ISNA(Rf_asReal(roldX)) && !ISNA(Rf_asReal(roldy))) {
        have_old_data = true;
        BOOM::Mat oldX(ToBoomMatrix(roldX));
        BOOM::Vec oldy(ToBoomVector(roldy));
        model->clear_data();
        for (int i = 0; i < oldy.size(); ++i) {
          NEW(RegressionData, dp)(oldy[i], oldX.row(i));
          model->add_data(dp);
        }
      } else {
        AddRegressionData(model, object);
      }

      io_manager.prepare_to_stream(object);
      int burn = lround(Rf_asReal(rburn));
      if (burn > 0) {
        io_manager.advance(burn);
      }

      int niter = LENGTH(getListElement(object, "sigma.obs"));

      PROTECT(forecast = Rf_allocMatrix(REALSXP, niter - burn, nrow(newX)));
      SubMatrix forecast_view(REAL(forecast), niter - burn, nrow(newX));
      if (burn < 0) {
        burn = 0;
      }
      for (int i = burn; i < niter; ++i) {
        io_manager.stream();
        if (have_old_data) {
          forecast_view.row(i - burn) = model->simulate_forecast(newX);
        } else {
          forecast_view.row(i - burn) = model->simulate_forecast(
              newX, final_state);
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
  // Compute one step ahead prediction errors from the Kalman filter.
  // Args:
  //   object: The R object returned by the R function bsts.  This is
  //     the object returned by fit_state_space_regression_, with
  //     other stuff added on by bsts.
  //   rnewX: An R matrix giving the predictor variables to be used
  //     in the forecast.
  //   rnewY: An R vector giving the values of the response in the
  //     holdout sample.
  //   rburn: An R scalar giving the number of MCMC iterations to be
  //     discarded before starting the forecast.
  // Returns:
  // An R matrix with rows corresponding to MCMC iterations, and
  // columns corresponding to forecast time.
  SEXP bsts_one_step_holdout_prediction_errors_(
      SEXP object,
      SEXP rnewX,
      SEXP rnewY,
      SEXP rburn) {
    RErrorReporter error_reporter;
    try {
      BOOM::Mat newX(ToBoomMatrix(rnewX));
      int xdim = ncol(newX);
      Ptr<StateSpaceRegressionModel> model(
          new StateSpaceRegressionModel(xdim));

      // Repopulate the training data so that the time index is set
      // correctly in the Kalman filter.
      AddRegressionData(model, object);

      Vec newY(ToBoomVector(rnewY));
      if (newX.nrow() != newY.size()) {
        error_reporter.SetError("error in bsts_one_step_prediction_errors_: "
                                "number of rows in X does not match length of "
                                "y");
        return R_NilValue;
      }

      Vec final_state(model->state_dimension());
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              &final_state,
              false,   // Don't save state contributions.
              false);  // Don't save prediction errors.

      io_manager.prepare_to_stream(object);
      int burn = lround(Rf_asReal(rburn));
      if (burn > 0) {
        io_manager.advance(burn);
      }

      int niter = LENGTH(getListElement(object, "sigma.obs"));
      SEXP errors;
      PROTECT(errors = Rf_allocMatrix(REALSXP, niter - burn, nrow(newX)));
      SubMatrix error_view(REAL(errors), niter - burn, nrow(newX));
      if (burn < 0) {
        burn = 0;
      }
      for (int i = burn; i < niter; ++i) {
        io_manager.stream();
        error_view.row(i - burn) = model->one_step_holdout_prediction_errors(
            newX, newY, final_state);
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

  //======================================================================
  // Compute one step ahead prediction errors from the Kalman filter,
  // for the training data.
  // Args:
  //   object: The R object returned by the R function bsts.  This is
  //     the object returned by fit_state_space_regression_, with
  //     other stuff added on by bsts.
  //   rburn: An R scalar giving the number of MCMC iterations to be
  //     discarded before starting the forecast.
  // Returns:
  //   An R matrix with rows corresponding to MCMC iterations, and
  //   columns corresponding to forecast time.
  SEXP bsts_one_step_training_prediction_errors_(SEXP object, SEXP rburn) {
    try {
      BOOM::Mat X(ToBoomMatrix(getListElement(object, "design.matrix")));
      int xdim = ncol(X);
      int time_dimension(nrow(X));
      Ptr<StateSpaceRegressionModel> model(new StateSpaceRegressionModel(xdim));

      // Repopulate the training data so that the time index is set
      // correctly in the Kalman filter.
      AddRegressionData(model, object);

      Vec final_state;
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              &final_state,
              false,   // Don't save state contributions.
              false);  // Don't save prediction errors.

      io_manager.prepare_to_stream(object);
      int burn = lround(Rf_asReal(rburn));
      if (burn > 0) {
        io_manager.advance(burn);
      }
      if (burn < 0) {
        burn = 0;
      }

      int niter = LENGTH(getListElement(object, "sigma.obs"));
      SEXP errors;
      PROTECT(errors = Rf_allocMatrix(REALSXP, niter - burn, time_dimension));
      SubMatrix error_view(REAL(errors), niter - burn, time_dimension);
      for (int i = burn; i < niter; ++i) {
        io_manager.stream();
        error_view.row(i - burn) = model->one_step_prediction_errors();
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
}  // extern "C"

}  // namespace bsts
