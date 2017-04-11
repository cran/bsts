#ifndef ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_
#define ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_

#include "model_manager.h"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"

namespace BOOM {
namespace bsts {
// This file contains two model factories, because there are separate
// implementations for Gaussian state space models with and without
// regression components.

class GaussianModelManagerBase : public ModelManager {
  StateSpaceModelBase * CreateModel(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_prior,
      SEXP r_options,
      Vector *final_state,
      bool save_state_contribution,
      bool save_prediction_errors,
      RListIoManager *io_manager) override;
};

class StateSpaceModelManager
    : public GaussianModelManagerBase {
 public:
  // Creates the model_ object, assigns a PosteriorSamper, and
  // allocates space in the io_manager for the objects in the
  // observation model.
  // Args:
  //   r_data_list: Contains a numeric vector named 'response' and a
  //     logical vector 'response.is.observed.'
  //   r_prior:  An R object of class SdPrior.
  //   r_options:  Not used.
  //   io_manager:  The io_manager that will record the MCMC draws.
  StateSpaceModel * CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) override;

  void AddDataFromBstsObject(SEXP r_bsts_object) override;
  void AddDataFromList(SEXP r_data_list) override;
  int UnpackForecastData(SEXP r_prediction_data) override;
  Vector SimulateForecast(const Vector &final_state) override;

  int UnpackHoldoutData(SEXP r_holdout_data) override;
  Vector HoldoutDataOneStepHoldoutPredictionErrors(
      const Vector &final_state) override;

 private:
  void AddData(const Vector &response,
               const std::vector<bool> &response_is_observed);

  Ptr<StateSpaceModel> model_;
  int forecast_horizon_;
  Vector holdout_data_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_
