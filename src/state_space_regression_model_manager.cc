#include "state_space_regression_model_manager.h"
#include "state_space_gaussian_model_manager.h"
#include "utils.h"

#include "r_interface/list_io.hpp"
#include "r_interface/prior_specification.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceRegressionSampler.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
namespace bsts {

namespace {
typedef StateSpaceRegressionModelManager SSRMF;
}  // namespace

StateSpaceRegressionModelManager::StateSpaceRegressionModelManager()
    : predictor_dimension_(-1) {}

StateSpaceRegressionModel * SSRMF::CreateObservationModel(
    SEXP r_data_list,
    SEXP r_prior,
    SEXP r_options,
    RListIoManager *io_manager) {

  if (!Rf_isNull(r_data_list)) {
    // If we were passed data from R then use it to build the model.
    Matrix predictors(ToBoomMatrix(getListElement(r_data_list, "predictors")));
    Vector response(ToBoomVector(getListElement(r_data_list, "response")));
    std::vector<bool> response_is_observed(ToVectorBool(getListElement(
        r_data_list, "response.is.observed")));
    UnpackTimestampInfo(r_data_list);
    if (TimestampsAreTrivial()) {
      model_.reset(new StateSpaceRegressionModel(
          response,
          predictors,
          response_is_observed));
    } else {
      // timestamps are non-trivial.
      model_.reset(new StateSpaceRegressionModel(ncol(predictors)));
      std::vector<Ptr<StateSpace::MultiplexedRegressionData>> data;
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        data.push_back(new StateSpace::MultiplexedRegressionData);
      }
      for (int i = 0; i < response.size(); ++i) {
        NEW(RegressionData, observation)(response[i], predictors.row(i));
        if (!response_is_observed[i]) {
          observation->set_missing_status(Data::completely_missing);
        }
        data[TimestampMapping(i)]->add_data(observation);
      }
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        if (data[i]->sample_size() == 0) {
          data[i]->set_missing_status(Data::completely_missing);
        }
        model_->add_data(data[i]);
      }
    }
  } else {
    // If no data was passed from R then build the model from its
    // default constructor.  We need to know the dimension of the
    // predictors.
    if (predictor_dimension_ < 0) {
      report_error("If r_data_list is not passed, you must call "
                   "SetPredictorDimension before calling "
                   "CreateObservationModel.");
    }
    model_.reset(new StateSpaceRegressionModel(predictor_dimension_));
  }

  // A NULL r_prior signals that no posterior sampler is needed.
  if (!Rf_isNull(r_prior)) {
    SetRegressionSampler(r_prior, r_options);
    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(model_.get()));
    model_->set_method(sampler);
  }

  // Make the io_manager aware of the model parameters.
  Ptr<RegressionModel> regression(model_->regression_model());
  io_manager->add_list_element(
      new GlmCoefsListElement(regression->coef_prm(), "coefficients"));
  io_manager->add_list_element(
      new StandardDeviationListElement(regression->Sigsq_prm(),
                                       "sigma.obs"));
  return model_.get();
}

void SSRMF::AddDataFromBstsObject(SEXP r_bsts_object) {
  AddData(ToBoomVector(getListElement(r_bsts_object, "original.series")),
          ToBoomMatrix(getListElement(r_bsts_object, "predictors")),
          IsObserved(getListElement(r_bsts_object, "original.series")));
}

void SSRMF::AddDataFromList(SEXP r_observed_data) {
  AddData(ToBoomVector(getListElement(r_observed_data, "original.series")),
          ToBoomMatrix(getListElement(r_observed_data, "predictors")),
          ToVectorBool(getListElement(r_observed_data,
                                      "response.is.observed")));
}

int SSRMF::UnpackForecastData(SEXP r_prediction_data) {
  forecast_predictors_ = ToBoomMatrix(getListElement(
      r_prediction_data, "predictors"));
  return forecast_predictors_.nrow();
}

Vector SSRMF::SimulateForecast(const Vector &final_state) {
  return model_->simulate_forecast(forecast_predictors_, final_state);
}

int SSRMF::UnpackHoldoutData(SEXP r_holdout_data) {
  holdout_responses_ = ToBoomVector(getListElement(
      r_holdout_data, "response"));
  forecast_predictors_ = ToBoomMatrix(getListElement(
      r_holdout_data, "predictors"));
  if (nrow(forecast_predictors_) != holdout_responses_.size()) {
    report_error("Data sizes do not align ");
  }
  return holdout_responses_.size();
}

Vector SSRMF::HoldoutDataOneStepHoldoutPredictionErrors(
    const Vector &final_state) {
  return model_->one_step_holdout_prediction_errors(
      forecast_predictors_, holdout_responses_, final_state);
}

void SSRMF::SetRegressionSampler(SEXP r_regression_prior,
                                 SEXP r_options) {
  // If either the prior object or the bma method is NULL then take
  // that as a signal the model is not being specified for the
  // purposes of MCMC, and bail out.
  if (Rf_isNull(r_regression_prior)
      || Rf_isNull(r_options)
      || Rf_isNull(getListElement(r_options, "bma.method"))) {
    return;
  }
  std::string bma_method = BOOM::ToString(getListElement(
      r_options, "bma.method"));
  if (bma_method == "SSVS") {
    SetSsvsRegressionSampler(r_regression_prior);
  } else if (bma_method == "ODA") {
    SetOdaRegressionSampler(r_regression_prior, r_options);
  } else {
    std::ostringstream err;
    err << "Unrecognized value of bma_method: " << bma_method;
    BOOM::report_error(err.str());
  }
}

void SSRMF::SetSsvsRegressionSampler(SEXP r_regression_prior) {
  BOOM::RInterface::RegressionConjugateSpikeSlabPrior prior(
      r_regression_prior, model_->regression_model()->Sigsq_prm());
  DropUnforcedCoefficients(model_->regression_model(),
                           prior.prior_inclusion_probabilities());
  Ptr<BregVsSampler> sampler(new BregVsSampler(
      model_->regression_model().get(),
      prior.slab(),
      prior.siginv_prior(),
      prior.spike()));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  int max_flips = prior.max_flips();
  if (max_flips > 0) {
    sampler->limit_model_selection(max_flips);
  }
  model_->regression_model()->set_method(sampler);
}

void SSRMF::SetOdaRegressionSampler(SEXP r_regression_prior,
                                    SEXP r_options) {
  SEXP r_oda_options = getListElement(r_options, "oda.options");
  BOOM::RInterface::IndependentRegressionSpikeSlabPrior prior(
      r_regression_prior, model_->regression_model()->Sigsq_prm());
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
          model_->regression_model().get(),
          prior.slab(),
          prior.siginv_prior(),
          prior.prior_inclusion_probabilities(),
          eigenvalue_fudge_factor,
          fallback_probability));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  DropUnforcedCoefficients(model_->regression_model(),
                           prior.prior_inclusion_probabilities());
  model_->regression_model()->set_method(sampler);
}

void StateSpaceRegressionModelManager::SetPredictorDimension(int xdim) {
  predictor_dimension_ = xdim;
}

void StateSpaceRegressionModelManager::AddData(
    const Vector &response,
    const Matrix &predictors,
    const std::vector<bool> &response_is_observed) {
  if (nrow(predictors) != response.size()
      || response_is_observed.size() != response.size()) {
    std::ostringstream err;
    err << "Argument sizes do not match in "
        << "StateSpaceRegressionModelManager::AddData" << endl
        << "nrow(predictors) = " << nrow(predictors) << endl
        << "response.size()  = " << response.size() << endl
        << "observed.size()  = " << response_is_observed.size();
    report_error(err.str());
  }

  for (int i = 0; i < response.size(); ++i) {
    Ptr<RegressionData> dp(new RegressionData(response[i], predictors.row(i)));
    if (!response_is_observed[i]) {
      dp->set_missing_status(Data::partly_missing);
    }
    model_->add_data(dp);
  }
}

}  // namespace bsts
}  // namespace BOOM
