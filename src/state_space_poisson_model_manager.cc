#include "state_space_poisson_model_manager.h"
#include "utils.h"

#include "r_interface/prior_specification.hpp"

#include "Models/StateSpace/PosteriorSamplers/StateSpacePoissonPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include "Models/MvnModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

namespace BOOM {
namespace bsts {

namespace {
typedef StateSpacePoissonModelManager SSPMM;
}

SSPMM::StateSpacePoissonModelManager()
    : predictor_dimension_(-1) {}

StateSpacePoissonModel * SSPMM::CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) {
  if (!Rf_isNull(r_data_list)) {
    // If we were passed data from R then build the model using the
    // data that was passed.
    bool regression = !Rf_isNull(getListElement(r_data_list, "predictors"));
    Vector counts(ToBoomVector(getListElement(r_data_list, "response")));
    Vector exposure(ToBoomVector(getListElement(r_data_list, "exposure")));
    // If there are no predictors then make an intercept.
    Matrix predictors =
        regression ?
        ToBoomMatrix(getListElement(r_data_list, "predictors")) :
        Matrix(counts.size(), 1.0);
    std::vector<bool> response_is_observed(ToVectorBool(getListElement(
        r_data_list, "response.is.observed")));
    model_.reset(
        new StateSpacePoissonModel(
            counts,
            exposure,
            predictors,
            response_is_observed));
    // With the Gaussian models we have two separate classes for the
    // regression and non-regression cases.  For non-Gaussian models
    // we have a single class with a regression bit that can be turned
    // off.
    model_->set_regression_flag(regression);
  } else {
    // If no data was passed from R then build the model from its
    // default constructor.  We need to know the dimension of the
    // predictors.
    if (predictor_dimension_ < 0) {
      report_error("If r_data_list is NULL then you must call "
                   "SetPredictorDimension before calling CreateModel.");
    }
    model_.reset(new StateSpacePoissonModel(predictor_dimension_));
  }

  Ptr<PoissonRegressionSpikeSlabSampler> observation_model_sampler;
  if (!Rf_isNull(r_prior)
      && Rf_inherits(r_prior, "SpikeSlabPriorBase")) {
    // If r_prior is NULL it could either mean that there are no
    // predictors, or that an existing model is being reinstantiated.
    RInterface::SpikeSlabGlmPrior prior_spec(r_prior);
    observation_model_sampler = new PoissonRegressionSpikeSlabSampler(
        model_->observation_model(),
        prior_spec.slab(),
        prior_spec.spike());
    DropUnforcedCoefficients(
        model_->observation_model(),
        prior_spec.spike()->prior_inclusion_probabilities());
    // Restrict number of model selection sweeps if max_flips was set.
    int max_flips = prior_spec.max_flips();
    if (max_flips > 0) {
      observation_model_sampler->limit_model_selection(max_flips);
    }
    // Make the io_manager aware of the model parameters.
    io_manager->add_list_element(
        new GlmCoefsListElement(model_->observation_model()->coef_prm(),
                                "coefficients"));
  } else {
    // In the non-regression (or no sampler necessary) case make a
    // spike and slab prior that never includes anything.
    observation_model_sampler = new PoissonRegressionSpikeSlabSampler(
        model_->observation_model(),
        new MvnModel(1),
        new VariableSelectionPrior(1, 0.0));
  }
  // Both the observation_model and the actual model_ need to have
  // their posterior samplers set.
  model_->observation_model()->set_method(observation_model_sampler);
  Ptr<StateSpacePoissonPosteriorSampler> sampler(
      new StateSpacePoissonPosteriorSampler(
          model_.get(),
          observation_model_sampler));
  model_->set_method(sampler);
  return model_.get();
}

void SSPMM::AddDataFromBstsObject(SEXP r_bsts_object) {
  SEXP r_counts = getListElement(r_bsts_object, "original.series");
  Vector counts = ToBoomVector(r_counts);
  AddData(counts,
          ToBoomVector(getListElement(r_bsts_object, "exposure")),
          ExtractPredictors(r_bsts_object, "predictors", counts.size()),
          IsObserved(r_counts));
}

void SSPMM::AddDataFromList(SEXP r_data_list) {
  SEXP r_counts = getListElement(r_data_list, "response");
  Vector counts = ToBoomVector(r_counts);
  AddData(counts,
          ToBoomVector(getListElement(r_data_list, "exposure")),
          ExtractPredictors(r_data_list, "predictors", counts.size()),
          ToVectorBool(getListElement(r_data_list, "response.is.observed")));
}

int SSPMM::UnpackForecastData(SEXP r_prediction_data) {
  forecast_exposure_ = ToBoomVector(getListElement(
      r_prediction_data, "exposure"));
  int n = forecast_exposure_.size();
  forecast_predictors_ = ExtractPredictors(r_prediction_data, "predictors", n);
  return n;
}

Vector SSPMM::SimulateForecast(const Vector &final_state) {
  return model_->simulate_forecast(
      forecast_predictors_, forecast_exposure_, final_state);
}

int SSPMM::UnpackHoldoutData(SEXP r_holdout_data) {
  holdout_response_ = ToBoomVector(getListElement(
      r_holdout_data, "response"));
  forecast_predictors_ = ExtractPredictors(
      r_holdout_data, "predictors", holdout_response_.size());
  forecast_exposure_ = ToBoomVector(getListElement(
      r_holdout_data, "exposure"));
  if (forecast_exposure_.size() != holdout_response_.size()) {
    report_error("Data sizes do not match in holdout data.");
  }
  return holdout_response_.size();
}

Vector SSPMM::HoldoutDataOneStepHoldoutPredictionErrors(
    const Vector &final_state) {
  PoissonDataImputer data_imputer;
  return model_->one_step_holdout_prediction_errors(
      GlobalRng::rng,
      data_imputer,
      holdout_response_,
      forecast_exposure_,
      forecast_predictors_,
      final_state);
}

void SSPMM::SetPredictorDimension(int xdim) {
  predictor_dimension_ = xdim;
}

void SSPMM::AddData(const Vector &counts,
                    const Vector &exposure,
                    const Matrix &predictors,
                    const std::vector<bool> &is_observed) {
  for (int i = 0; i < counts.size(); ++i) {
    Ptr<StateSpace::AugmentedPoissonRegressionData> data_point(
        new StateSpace::AugmentedPoissonRegressionData(
            counts[i],
            exposure[i],
            predictors.row(i)));
    if (!is_observed.empty() && !is_observed[i]) {
      data_point->set_missing_status(Data::missing_status::completely_missing);
    }
    model_->add_data(data_point);
  }
}


}  // namespace bsts
}  // namespace BOOM
