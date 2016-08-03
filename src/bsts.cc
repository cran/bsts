#include <ctime>

#include "utils.h"
#include "model_manager.h"

#include "r_interface/check_interrupt.h"
#include "r_interface/error.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "cpputil/report_error.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

extern "C" {
using BOOM::Vector;
using BOOM::Matrix;
using BOOM::Ptr;
using BOOM::bsts::ModelManager;
using BOOM::RCheckInterrupt;

SEXP fit_bsts_model_(
    SEXP r_data_list,
    SEXP r_state_specification,
    SEXP r_prior,
    SEXP r_options,
    SEXP r_family,
    SEXP r_save_state_contribution_flag,
    SEXP r_save_prediction_errors_flag,
    SEXP r_niter,
    SEXP r_ping,
    SEXP r_timeout_in_seconds,
    SEXP r_seed) {
  BOOM::RErrorReporter error_reporter;
  BOOM::RMemoryProtector protector;
  try {
    BOOM::RInterface::seed_rng_from_R(r_seed);
    BOOM::RListIoManager io_manager;
    bool save_state_contribution = Rf_asLogical(
        r_save_state_contribution_flag);
    bool save_prediction_errors = Rf_asLogical(
        r_save_prediction_errors_flag);
    std::string family = BOOM::ToString(r_family);
    int xdim = 0;
    SEXP r_predictors = BOOM::getListElement(r_data_list, "predictors");
    if (!Rf_isNull(r_predictors)) {
      xdim = Rf_ncols(r_predictors);
    }
    std::unique_ptr<ModelManager> model_manager(ModelManager::Create(
        family, xdim));
    Ptr<BOOM::StateSpaceModelBase> model(model_manager->CreateModel(
        r_data_list,
        r_state_specification,
        r_prior,
        r_options,
        nullptr,
        save_state_contribution,
        save_prediction_errors,
        &io_manager));

    // Do one posterior sampling step before getting ready to write.
    // This will ensure that any dynamically allocated objects have
    // the correct size before any R memory gets allocated in the
    // call to prepare_to_write().
    model->sample_posterior();
    int niter = lround(Rf_asReal(r_niter));
    int ping = lround(Rf_asReal(r_ping));
    double timeout_threshold_seconds = Rf_asReal(r_timeout_in_seconds);
    SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
    clock_t start_time = clock();
    double time_threshold = CLOCKS_PER_SEC * timeout_threshold_seconds;
    for (int i = 0; i < niter; ++i) {
      if (RCheckInterrupt()) {
        error_reporter.SetError("Canceled by user.");
        return R_NilValue;
      }
      BOOM::print_R_timestamp(i, ping);
      try {
        model->sample_posterior();
        io_manager.write();
        clock_t current_time = clock();
        if (current_time - start_time > time_threshold) {
          std::ostringstream warning;
          warning << "Timeout threshold "
                  << time_threshold
                  << " exceeded in iteration " << i << "."
                  << std::endl
                  << "Time used was "
                  << double(current_time - start_time) / CLOCKS_PER_SEC
                  << " seconds.";
          Rf_warning(warning.str().c_str());
          return BOOM::appendListElement(
              ans,
              ToRVector(BOOM::Vector(1, i + 1)),
              "ngood");
        }
      } catch(std::exception &e) {
        std::ostringstream err;
        err << "Caught an exception with the following "
            << "error message in MCMC "
            << "iteration " << i << ".  Aborting." << std::endl
            << e.what() << std::endl;
        error_reporter.SetError(err.str());
        return BOOM::appendListElement(ans,
                                       ToRVector(Vector(1, i)),
                                       "ngood");
      }
    }
    return ans;
  } catch (std::exception &e) {
    BOOM::RInterface::handle_exception(e);
  } catch (...) {
    BOOM::RInterface::handle_unknown_exception();
  }
  return R_NilValue;
}

// Returns the posterior predictive distribution of a model forecast
// over a specified forecast period.
// Args:
//   r_bsts_object: The object on which the predictions are to be
//     based, which was returned by the original call to bsts.
//   r_prediction_data: An R list containing any additional data
//     needed to make the prediction.  For simple state space models
//     this is just an integer giving the time horizon over which to
//     predict.  For models containing a regression component it
//     contains the future values of the X's.  For binomial (or
//     Poisson) models it contains a sequence of future trial counts
//     (or exposures).
//   r_burn: An integer giving the number of burn-in iterations to
//     discard.  Negative numbers will be treated as zero.  Numbers
//     greater than the number of MCMC iterations will raise an error.
//   r_observed_data: An R list containing the observed data on which
//     to base the prediction.  In typical cases this should be
//     R_NilValue (R's NULL) signaling that the observed data should
//     be taken from r_bsts_object.  However, if additional data have
//     been observed since the model was trained, or if the model is
//     being used to predict data that were part of the training set,
//     or some other application other than making predictions
//     starting from one time period after the training data ends,
//     then one can use this argument to pass the "training data" on
//     which the predictions should be based.  If this argument is
//     used, then the Kalman filter will be run over the supplied
//     data, which is expensive.  If this argument is left as
//     R_NilValue (NULL) then the "final.state" element of
//     r_bsts_object will be used as the basis for future predictions,
//     which is a computational savings over filtering from scratch.
//
// Returns:
//   An R matrix containing draws from the posterior predictive
//   distribution.  Rows of the matrix correspond to MCMC iterations,
//   and columns to time points.  The matrix will have 'burn' fewer
//   rows than the number of MCMC iterations in r_bsts_object.
SEXP predict_bsts_model_(
    SEXP r_bsts_object,
    SEXP r_prediction_data,
    SEXP r_burn,
    SEXP r_observed_data) {
  try {
    std::unique_ptr<ModelManager> model_manager(
        ModelManager::Create(r_bsts_object));
    return BOOM::ToRMatrix(model_manager->Forecast(
        r_bsts_object,
        r_prediction_data,
        r_burn,
        r_observed_data));
  } catch (std::exception &e) {
    BOOM::RInterface::handle_exception(e);
  } catch (...) {
    BOOM::RInterface::handle_unknown_exception();
  }
  return R_NilValue;
}

// Compute the distribution of one-step prediction errors for the
// training data or a set of holdout data.
//
// Args:
//   r_bsts_object: The object on which the predictions are to be
//     based, which was returned by the original call to bsts.
//   r_holdout_data: This can be R_NilValue, in which case the one
//     step prediction errors from the training data are computed.  Or
//     it can be a set of data formatted as in the r_data_list
//     argument to fit_bsts_model_.  If the latter,
//     then it is assumed to be a holdout data set that takes place
//     immediately after the last observation in the training data.
//   r_burn: An integer giving the number of burn-in iterations to
//     discard.  Negative numbers will be treated as zero.  Numbers
//     greater than the number of MCMC iterations will raise an error.
//
// Returns:
//    An R matrix with rows corresponding to MCMC draws and columns
//    corresponding to time.  If a holdout data set is supplied then
//    the number of columns in the matrix matches the number of
//    observations in the holdout data.  Otherwise it matches the
//    number of observations in the training data.
SEXP bsts_one_step_prediction_errors_(
    SEXP r_bsts_object,
    SEXP r_holdout_data,
    SEXP r_burn) {
  try {
    std::unique_ptr<ModelManager> model_manager(
        ModelManager::Create(r_bsts_object));
    return BOOM::ToRMatrix(model_manager->OneStepPredictionErrors(
        r_bsts_object,
        r_holdout_data,
        r_burn));
  } catch (std::exception &e) {
    BOOM::RInterface::handle_exception(e);
  } catch (...) {
    BOOM::RInterface::handle_unknown_exception();
  }
  return R_NilValue;
}

}  // extern "C"
