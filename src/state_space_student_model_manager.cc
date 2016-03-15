#include "state_space_student_model_manager.h"
#include "utils.h"

#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceStudentPosteriorSampler.hpp"
#include "r_interface/prior_specification.hpp"


namespace BOOM {
namespace bsts {

namespace {
typedef StateSpaceStudentModelManager SSSMM;
}

SSSMM::StateSpaceStudentModelManager()
    : predictor_dimension_(-1) {}

StateSpaceStudentRegressionModel * SSSMM::CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) {
  if (!Rf_isNull(r_data_list)) {
    // If we were passed data from R then use it to build the model.
    SEXP r_predictors = getListElement(r_data_list, "predictors");
    bool regression = !Rf_isNull(r_predictors);
    Vector response(ToBoomVector(getListElement(r_data_list, "response")));
    // If there are no predictors then make an intercept.
    Matrix predictors =
        regression ?
        ToBoomMatrix(r_predictors) :
        Matrix(response.size(), 1, 1.0);
    std::vector<bool> response_is_observed(ToVectorBool(getListElement(
        r_data_list, "response.is.observed")));
    model_.reset(new StateSpaceStudentRegressionModel(
        response, predictors, response_is_observed));
    model_->set_regression_flag(regression);
  } else {
    // If no data was passed from R then build the model from its
    // default constructor.  We need to know the dimension of the
    // predictors.
    if (predictor_dimension_ < 0) {
      report_error("If r_data_list is NULL then you must call "
                   "SetPredictorDimension before creating a model.");
    }
    model_.reset(new StateSpaceStudentRegressionModel(predictor_dimension_));
  }

  // A NULL r_prior signals that no posterior sampler is needed.  This
  // differes from the logit and Poisson cases, where a NULL prior
  // might signal the absence of predictors, because the T model still
  // needs a prior for the residual "variance" and tail thickness
  // parameters.
  if (!Rf_isNull(r_prior)) {
    TRegressionModel *regression = model_->observation_model();
    BOOM::RInterface::StudentRegressionConjugateSpikeSlabPrior prior_spec(
        r_prior, regression->Sigsq_prm());
    Ptr<TRegressionSpikeSlabSampler> observation_model_sampler(
        new TRegressionSpikeSlabSampler(
            regression,
            prior_spec.slab(),
            prior_spec.spike(),
            prior_spec.siginv_prior(),
            prior_spec.degrees_of_freedom_prior()));
    DropUnforcedCoefficients(regression,
                             prior_spec.prior_inclusion_probabilities());
    // Restrict number of attempted flips and the domain of the
    // residual "standard deviation" if these have been set.
    observation_model_sampler->set_sigma_upper_limit(
        prior_spec.sigma_upper_limit());
    int max_flips = prior_spec.max_flips();
    if (max_flips > 0) {
      observation_model_sampler->limit_model_selection(max_flips);
    }
    // Both the observation_model and the actual model_ need to have
    // their posterior samplers set.
    regression->set_method(observation_model_sampler);
    Ptr<StateSpaceStudentPosteriorSampler> sampler(
        new StateSpaceStudentPosteriorSampler(
            model_.get(),
            observation_model_sampler));
    model_->set_method(sampler);
  }

  // Make the io_manager aware of all the model parameters.
  if (model_->observation_model()->xdim() > 1) {
    io_manager->add_list_element(
        new GlmCoefsListElement(
            model_->observation_model()->coef_prm(),
            "coefficients"));
  }
  io_manager->add_list_element(
      new StandardDeviationListElement(
          model_->observation_model()->Sigsq_prm(),
          "sigma.obs"));
  io_manager->add_list_element(
      new UnivariateListElement(
          model_->observation_model()->Nu_prm(),
          "observation.df"));

  return model_.get();
}

void SSSMM::AddDataFromBstsObject(SEXP r_bsts_object) {
  SEXP r_response = getListElement(r_bsts_object, "original.series");
  Vector response = ToBoomVector(r_response);
  AddData(response,
          ExtractPredictors(r_bsts_object, "predictors", response.size()),
          IsObserved(r_response));
}

void SSSMM::AddDataFromList(SEXP r_data_list) {
  Vector response = ToBoomVector(getListElement(r_data_list, "response"));
  AddData(response,
          ExtractPredictors(r_data_list, "predictors", response.size()),
          ToVectorBool(getListElement(r_data_list, "response.is.observed")));
}

int SSSMM::UnpackForecastData(SEXP r_prediction_data) {
  SEXP r_horizon = getListElement(r_prediction_data, "horizon");
  if (Rf_isNull(r_horizon)) {
    forecast_predictors_ = ToBoomMatrix(getListElement(
        r_prediction_data, "predictors"));
  } else {
    forecast_predictors_ = Matrix(Rf_asInteger(r_horizon), 1, 1.0);
  }
  return forecast_predictors_.nrow();
}

Vector SSSMM::SimulateForecast(const Vector &final_state) {
  return model_->simulate_forecast(forecast_predictors_,
                                   final_state);
}

int SSSMM::UnpackHoldoutData(SEXP r_holdout_data) {
  holdout_response_ = ToBoomVector(getListElement(r_holdout_data, "response"));
  forecast_predictors_ = ExtractPredictors(
      r_holdout_data, "predictors", holdout_response_.size());
  return holdout_response_.size();
}

Vector SSSMM::HoldoutDataOneStepHoldoutPredictionErrors(
    const Vector &final_state) {
  return model_->one_step_holdout_prediction_errors(
      GlobalRng::rng,
      holdout_response_,
      forecast_predictors_,
      final_state);
}

void SSSMM::AddData(const Vector &response,
                    const Matrix &predictors,
                    const std::vector<bool> &response_is_observed) {
  int sample_size = response.size();
  for (int i = 0; i < sample_size; ++i) {
    Ptr<StateSpace::VarianceAugmentedRegressionData> data_point(
        new StateSpace::VarianceAugmentedRegressionData(
            response[i],
            predictors.row(i)));
    if (!response_is_observed.empty() && !response_is_observed[i]) {
      data_point->set_missing_status(Data::missing_status::completely_missing);
    }
    model_->add_data(data_point);
  }
}

void SSSMM::SetPredictorDimension(int xdim) {
  predictor_dimension_ = xdim;
}

}  // namespace bsts
}  // namespace BOOM
