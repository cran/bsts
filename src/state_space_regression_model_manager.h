#ifndef BSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_
#define BSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_

#include "state_space_gaussian_model_manager.h"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"

namespace BOOM {
namespace bsts {

class StateSpaceRegressionModelManager
    : public GaussianModelManagerBase {
 public:
  StateSpaceRegressionModelManager();

  StateSpaceRegressionModel * CreateObservationModel(
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

  void SetPredictorDimension(int xdim);

 private:
  // Assign a prior distribution and posterior sampler to the
  // observation_model portion of a StateSpaceRegressionModel.
  //
  // Args:
  //   r_regression_prior: An R object encoding the prior distribution
  //     to use for the model.  This argument can be NULL if the model
  //     is not being created for the purposes of learning (e.g. if an
  //     already learned model is being reinstantiated for purposes of
  //     forecasting or diagnostics).
  //   r_options: A list that must contain the following elements if the
  //     object is being constructed for learning.
  //     * bma.method: An R string specifying whether "SSVS"
  //         (stochastic search variable selection: George and
  //         McCulloch 1997 statistica sinica) or "ODA" (orthoganal
  //         data augmentation, Ghosh and Clyde 2012 JASA) should be
  //         used for Bayesian model averaging.  Can also be
  //         R_NilValue if the model is not being specified for MCMC.
  //     * oda.options: If the bma.method == "ODA" then this is a list
  //         containing "eigenvalue_fudge_factor" and
  //         "fallback_probability".  See the bsts documentation for
  //         details.
  void SetRegressionSampler(SEXP r_regression_prior, SEXP r_options);
  void SetSsvsRegressionSampler(SEXP r_regression_prior);
  void SetOdaRegressionSampler(SEXP r_regression_prior, SEXP r_options);

  void AddData(const Vector &response,
               const Matrix &predictors,
               const std::vector<bool> &response_is_observed);

  Ptr<StateSpaceRegressionModel> model_;
  int predictor_dimension_;
  Matrix forecast_predictors_;
  Vector holdout_responses_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_STATE_SPACE_REGRESSION_MODEL_MANAGER_H_
