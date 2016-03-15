#ifndef BSTS_SRC_STATE_SPACE_STUDENT_MODEL_MANAGER_H_
#define BSTS_SRC_STATE_SPACE_STUDENT_MODEL_MANAGER_H_

#include "model_manager.h"
#include "Models/StateSpace/StateSpaceStudentRegressionModel.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"

namespace BOOM {
namespace bsts {

class StateSpaceStudentModelManager
    : public ModelManager {
 public:
  StateSpaceStudentModelManager();

  StateSpaceStudentRegressionModel * CreateObservationModel(
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
  void AddData(const Vector &response,
               const Matrix &predictors,
               const std::vector<bool> &response_is_observed);

  Ptr<StateSpaceStudentRegressionModel> model_;
  int predictor_dimension_;

  Matrix forecast_predictors_;
  Vector holdout_response_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_STATE_SPACE_STUDENT_MODEL_MANAGER_H_
