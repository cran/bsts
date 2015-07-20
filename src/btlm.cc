// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include "utils.h"

#include "r_interface/check_interrupt.h"
#include "r_interface/error.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "cpputil/report_error.hpp"

#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceRegressionSampler.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"

#include "R_ext/Arith.h"  // for ISNA

using BOOM::BregVsSampler;
using BOOM::SpikeSlabDaRegressionSampler;
using BOOM::Matrix;
using BOOM::Ptr;
using BOOM::SpdMatrix;
using BOOM::SubMatrix;
using BOOM::Vector;

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
void AddRegressionPriorAndSetSampler(
    Ptr<StateSpaceRegressionModel> model,
    SEXP r_regression_prior,
    SEXP r_bma_method,
    SEXP r_oda_options);

void AddRegressionData(Ptr<StateSpaceRegressionModel>, SEXP bsts_object);

RListIoManager SpecifyStateSpaceRegressionModel(
    Ptr<StateSpaceRegressionModel> model,
    SEXP r_state_specification,
    SEXP r_regression_prior,
    SEXP r_bma_method,
    SEXP r_oda_options,
    Vector *final_state,
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
  //
  // TODO(stevescott): Change the leading 'virtual' to a trailing
  // 'override' once CRAN supports C++11 on windows.
  virtual int nrow() const {return 1 + model_->nstate();}
  virtual int ncol() const {return model_->time_dimension();}
  virtual BOOM::Matrix get_matrix() const {
    BOOM::Matrix ans(nrow(), ncol());
    for (int state = 0; state < model_->nstate(); ++state) {
      ans.row(state) = model_->state_contribution(state);
    }
    ans.last_row() = model_->regression_contribution();
    return ans;
  }

 private:
  BOOM::StateSpaceRegressionModel *model_;
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
//   bma_method: An R string specifying whether "SSVS" or "ODA" should
//     be used for Bayesian model averaging.  Can be an R NULL value
//     if no MCMC is needed.
//   final_state: A pointer to a BOOM::Vector that can be used to read in
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
    SEXP r_state_specification,
    SEXP r_regression_prior,
    SEXP r_bma_method,
    SEXP r_oda_options,
    Vector *final_state,
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
  factory.AddState(r_state_specification);
  AddRegressionPriorAndSetSampler(model,
                                  r_regression_prior,
                                  r_bma_method,
                                  r_oda_options);
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

// Initialize the model to be empty, except for variables that are
// known to be present with probability 1.
void DropAllCoefficients(Ptr<RegressionModel> regression,
                         const BOOM::Vector &prior_inclusion_probs) {
  regression->coef().drop_all();
  for (int i = 0; i < prior_inclusion_probs.size(); ++i) {
    if (prior_inclusion_probs[i] >= 1.0) {
      regression->coef().add(i);
    }
  }
}

//======================================================================
// Adds a regression prior to the RegressionModel managed by model.
// Args:
//   model: The StateSpaceRegressionModel that needs a regression
//     prior assigned to it.
//   regression_prior: An R object created by SpikeSlabPrior,
//     which is part of the BoomSpikeSlab package.
//   bma_method: An R string specifying whether "SSVS" or "ODA" should
//     be used for Bayesian model averaging.  Can also be R_NilValue
//     if the model is not being specified for MCMC.
void AddRegressionPriorAndSetSampler(Ptr<StateSpaceRegressionModel> model,
                                     SEXP r_regression_prior,
                                     SEXP r_bma_method,
                                     SEXP r_oda_options) {
  // If either the prior object or the bma method is NULL then take
  // that as a signal the model is not being specified for the
  // purposes of MCMC, and bail out.
  if (Rf_isNull(r_bma_method) || Rf_isNull(r_regression_prior)) {
    return;
  }
  std::string bma_method = BOOM::ToString(r_bma_method);
  Ptr<RegressionModel> regression(model->regression_model());
  if (bma_method == "SSVS") {
    BOOM::RInterface::RegressionConjugateSpikeSlabPrior prior(
        r_regression_prior, model->regression_model()->Sigsq_prm());
    DropAllCoefficients(regression, prior.prior_inclusion_probabilities());
    Ptr<BregVsSampler> sampler(new BregVsSampler(
        model->regression_model().get(),
        prior.slab(),
        prior.siginv_prior(),
        prior.spike()));
    sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
    int max_flips = prior.max_flips();
    if (max_flips > 0) {
      sampler->limit_model_selection(max_flips);
    }
    regression->set_method(sampler);
  } else if (bma_method == "ODA") {
    BOOM::RInterface::IndependentRegressionSpikeSlabPrior prior(
        r_regression_prior, model->regression_model()->Sigsq_prm());
    double eigenvalue_fudge_factor = 0.001;
    double fallback_probability = 0.0;
    if (!Rf_isNull(r_oda_options)) {
      eigenvalue_fudge_factor = Rf_asReal(
          getListElement(r_oda_options, "eigenvalue.fudge.factor"));
      fallback_probability = Rf_asReal(
          getListElement(r_oda_options, "fallback.probability"));
    }
    Ptr<SpikeSlabDaRegressionSampler> sampler(
        new SpikeSlabDaRegressionSampler(
            regression.get(),
            prior.slab(),
            prior.siginv_prior(),
            prior.prior_inclusion_probabilities(),
            eigenvalue_fudge_factor,
            fallback_probability));
    sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
    regression->set_method(sampler);
  } else {
    std::ostringstream err;
    err << "Unrecognized value of bma_method: " << bma_method;
    BOOM::report_error(err.str());
  }
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
  Matrix X(ToBoomMatrix(getListElement(object, "design.matrix")));
  Vector y(ToBoomVector(getListElement(object, "original.series")));
  for (int i = 0; i < y.size(); ++i) {
    model->add_data(Ptr<RegressionData>(new RegressionData(y[i], X.row(i))));
  }
}

extern "C" {
  //======================================================================
  // Primary C++ interface to the R bsts function when using a formula.
  // Args:
  //   r_x: An R matrix created using model.matrix.  This is the design
  //     matrix for the regression.
  //   r_y: An R vector of responses created using model.response from
  //     the R function bsts().
  //   r_y_is_observed: An R logical vector indicating which elements
  //     of y are non-NA.  NA elements of y correspond to missing
  //     observations that should be smoothed over by the Kalman
  //     filter.
  //   r_state_specification: An R list of state model specifications.  See
  //     the R documentation and the comments in create_state_model.cc
  //   r_save_state_contribution_flag: An R logical indicating whether
  //     contributions from the individual state model components
  //     should be saved.  Saving them requires storage of order
  //     (number of time points) * (number of state models) * (number
  //     of mcmc iterations)
  //   r_regression_prior: An R object specifying the prior for the
  //     regression coefficients and observation variance.  This is
  //     assumed to have been created by SpikeSlabPrior, part of
  //     the BoomSpikeSlab package.
  //   r_bma_method: A string indicating which strategy to use for
  //     Bayesian model averaging.  The choices are "SSVS" and "ODA".
  //   r_oda_options: A list (which is ignored unless rbma_method ==
  //     "ODA") with the following elements:
  //     * fallback.probability: Each MCMC iteration will use SSVS
  //       instead of ODA with this probability.  In cases where the
  //       latent data have high leverage, ODA mixing can suffer.
  //       Mixing in a few SSVS steps can help keep an errant
  //       algorithm on track.
  //    * eigenvalue.fudge.factor: The latent X's will be chosen so
  //      that the complete data X'X matrix (after scaling) is a
  //      constant diagonal matrix equal to the largest eigenvalue of
  //      the observed (scaled) X'X times (1 +
  //      eigenvalue.fudge.factor).  This should be a small positive
  //      number.
  //   r_niter:  An R scalar giving the desired number of MCMC iterations.
  //   r_ping: An R integer indicating the frequency with which
  //     progress reports get printed.  E.g. setting rping = 100 will
  //     print a status message with a time and iteration stamp every
  //     100 iterations.  If you don't want these messages set rping < 0.
  //   r_seed: An integer to use as the C++ random seed, or NULL.  If
  //     NULL then the C++ seed will be set using the clock.
  // Returns:
  //   An R object containing the posterior draws defining the model.
  SEXP bsts_fit_state_space_regression_(
      SEXP r_x,
      SEXP r_y,
      SEXP r_y_is_observed,
      SEXP r_state_specification,
      SEXP r_save_state_contribution_flag,
      SEXP r_save_prediction_errors_flag,
      SEXP r_regression_prior,
      SEXP r_bma_method,
      SEXP r_oda_options,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed) {
    RErrorReporter error_reporter;
    try {
      // The try block is needed to catch any errors that come from
      // boom_r_tools.  E.g. ToBoomVector...
      BOOM::RInterface::seed_rng_from_R(r_seed);
      BOOM::Matrix X = ToBoomMatrix(r_x);
      BOOM::Vector y = ToBoomVector(r_y);

      std::vector<bool> y_is_observed = ToVectorBool(r_y_is_observed);

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
              r_state_specification,
              r_regression_prior,
              r_bma_method,
              r_oda_options,
              NULL,
              Rf_asLogical(r_save_state_contribution_flag),
              Rf_asLogical(r_save_prediction_errors_flag));

      // Do one posterior sampling step before getting ready to write.
      // This will ensure that any dynamically allocated objects have
      // the correct size before any R memory gets allocated in the
      // call to prepare_to_write().
      model->sample_posterior();

      int niter = lround(Rf_asReal(r_niter));
      int ping = lround(Rf_asReal(r_ping));
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
      BOOM::Matrix newX(ToBoomMatrix(rnewX));
      int xdim = ncol(newX);
      Ptr<StateSpaceRegressionModel> model(
          new StateSpaceRegressionModel(xdim));

      Vector final_state;
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              R_NilValue,  // bma method
              R_NilValue,  // oda options
              &final_state,
              false,       // Don't save state contributions.
              false);      // Don't save prediction errors.

      bool have_old_data(false);
      if (!ISNA(Rf_asReal(roldX)) && !ISNA(Rf_asReal(roldy))) {
        have_old_data = true;
        BOOM::Matrix oldX(ToBoomMatrix(roldX));
        BOOM::Vector oldy(ToBoomVector(roldy));
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
      BOOM::Matrix newX(ToBoomMatrix(rnewX));
      int xdim = ncol(newX);
      Ptr<StateSpaceRegressionModel> model(
          new StateSpaceRegressionModel(xdim));

      // Repopulate the training data so that the time index is set
      // correctly in the Kalman filter.
      AddRegressionData(model, object);

      Vector newY(ToBoomVector(rnewY));
      if (newX.nrow() != newY.size()) {
        error_reporter.SetError("error in bsts_one_step_prediction_errors_: "
                                "number of rows in X does not match length of "
                                "y");
        return R_NilValue;
      }

      Vector final_state(model->state_dimension());
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              R_NilValue,   // bma method
              R_NilValue,   // epsilon
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
      BOOM::Matrix X(ToBoomMatrix(getListElement(object, "design.matrix")));
      int xdim = ncol(X);
      int time_dimension(nrow(X));
      Ptr<StateSpaceRegressionModel> model(new StateSpaceRegressionModel(xdim));

      // Repopulate the training data so that the time index is set
      // correctly in the Kalman filter.
      AddRegressionData(model, object);

      Vector final_state;
      RListIoManager io_manager =
          SpecifyStateSpaceRegressionModel(
              model,
              getListElement(object, "state.specification"),
              getListElement(object, "regression.prior"),
              R_NilValue,  // bma method
              R_NilValue,  // epsilon
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
