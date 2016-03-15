#ifndef BSTS_SRC_MODEL_MANAGER_H_
#define BSTS_SRC_MODEL_MANAGER_H_

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {
namespace bsts {

// The job of a ModelManager is to construct the BOOM models that bsts
// uses for inference, and to provide any additional data that the
// models need for tasks other than statistical inference.
//
// The point of the ModelManager is to be an intermediary between the
// calls in bsts.cc and the underlying BOOM code, so that R can pass
// lists of data, priors, and model options formatted as expected by
// the child ModelManager classes for specific model types.
class ModelManager {
 public:
  virtual ~ModelManager() {}

  // Create a ModelManager instance suitable for working with the
  // specified family.
  // Args:
  //   family: A text string identifying the model family.
  //     "gaussian", "logit", "poisson", or "student".
  //   xdim: Dimension of the predictors in the observation model
  //     regression.  This can be zero if there are no regressors.
  static ModelManager * Create(const std::string &family, int xdim);

  // Create a model manager by reinstantiating a previously
  // constructed bsts model.
  // Args:
  //   r_bsts_object:  An object previously created by a call to bsts.
  static ModelManager * Create(SEXP r_bsts_object);

  // Creates a BOOM state space model suitable for learning with MCMC.
  // Args:
  //   r_data_list: An R list containing the data to be modeled in the
  //     format expected by the requested model family.  This list
  //     generally contains an object called 'response' and a logical
  //     vector named 'response.is.observed'.  If the model is a
  //     (generalized) regression model then it will contain a
  //     'predictors' object as well, otherwise 'predictors' will be
  //     NULL.  For logit or Poisson models an additional component
  //     should be present giving the number of trials or the
  //     exposure.
  //   r_state_specification: The R list created by the state
  //     configuration functions (e.g. AddLocalLinearTrend, AddSeasonal,
  //     etc).
  //   r_prior: The prior distribution for the observation model.  If
  //     the model is a regression model (determined by whether
  //     r_data_list contains a non-NULL 'predictors' element) then this
  //     must be some form of spike and slab prior.  Otherwise it is a
  //     prior for the error in the observation equation.  For single
  //     parameter error distributions like binomial or Poisson this can
  //     be NULL.
  //   r_options: Model or family specific options such as the technique
  //     to use for model averaging (ODA vs SVSS).
  //   family: A text string indicating the desired model family for the
  //     observation equation.
  //   save_state_contribution: A flag indicating whether the
  //     state-level contributions should be saved by the io_manager.
  //   save_prediction_errors: A flag indicating whether the one-step
  //     prediction errors from the Kalman filter should be saved by the
  //     io_manager.
  //   final_state: A pointer to a Vector to hold the state at the final
  //     time point.  This can be a nullptr if the state is only going
  //     to be recorded, but it must point to a Vector if the state is
  //     going to be read from an existing object.
  //   io_manager: The io_manager responsible for writing MCMC output to
  //     an R object, or streaming it from an existing object.
  //
  // Returns:
  //  A pointer to the created model.  The pointer is owned by a Ptr
  //  in the model manager, and should be caught by a Ptr in the caller.
  //
  // Side Effects:
  //   The returned pointer is also held in a smart pointer owned by
  //   the child class.
  virtual StateSpaceModelBase * CreateModel(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_prior,
      SEXP r_options,
      Vector *final_state,
      bool save_state_contribution,
      bool save_prediction_errors,
      RListIoManager *io_manager);

  // Returns a set of draws from the posterior predictive distribution.
  // Args:
  //   r_bsts_object:  The R object created from a previous call to bsts().
  //   r_prediction_data: Data needed to make the prediction.  This
  //     might be a data frame for models that have a regression
  //     component, or a vector of exposures or trials for binomial or
  //     Poisson data.
  //   r_options: If any special options need to be passed in order to
  //     do the prediction, they should be included here.
  //   r_observed_data: In most cases, the prediction takes place
  //     starting with the time period immediately following the last
  //     observation in the training data.  If so then r_observed_data
  //     should be R_NilValue, and the observed data will be taken
  //     from r_bsts_object.  However, if more data have been added
  //     (or if some data should be omitted) from the training data, a
  //     new set of training data can be passed here.
  //
  // Returns:
  //   An R matrix, with rows corresponding to MCMC draws and columns
  //   to time, containing posterior predictive draws for the
  //   forecast.
  virtual Matrix Forecast(
      SEXP r_bsts_object,
      SEXP r_prediction_data,
      SEXP r_burn,
      SEXP r_observed_data);

  // Returns the one step ahead prediction errors from the training
  // data.  In the typical case these were stored by the Kalman filter
  // in the original model fit, so this function will only rarely be
  // needed.
  //
  // Args:
  //   r_bsts_object: The R object created from a previous call to
  //     bsts().
  //   r_newdata: This can be R_NilValue, in which case the one step
  //     prediction errors from the training data are computed.  Or it
  //     can be a set of data formatted as in the r_data_list argument
  //     to CreateModel.  If the latter, then it is assumed to be a
  //     holdout data set that takes place immediately after the last
  //     observation in the training data.
  //   r_burn: An integer giving the desired number of MCMC iterations
  //     to discard.
  //
  // Returns:
  //    A matrix with rows corresponding to MCMC draws and columns
  //    corresponding to time.  If a holdout data set is supplied then
  //    the number of columns in the matrix matches the number of
  //    observations in the holdout data.  Otherwise it matches the
  //    number of observations in the training data.
  virtual Matrix OneStepPredictionErrors(
      SEXP r_bsts_object,
      SEXP r_newdata,
      SEXP r_burn);

 private:
  // Create the specific StateSpaceModel suitable for the given model
  // family.  The posterior sampler for the model is set, and entries
  // for its model parameters are created in io_manager.  This
  // function does not add state to the the model.  It is primarily
  // intended to aid the implementation of CreateModel.
  //
  // The arguments are documented in the comment to CreateModel.
  //
  // Returns:
  //   A pointer to the created model.  The pointer is owned by a Ptr
  //   in the the child class, so working with the raw pointer
  //   privately is exception safe.
  virtual StateSpaceModelBase * CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) = 0;

  // Add data to the model object managed by the child classes.  The
  // data can come either from a previous bsts object, or from an R
  // list containing appropriately formatted data.
  virtual void AddDataFromBstsObject(SEXP r_bsts_object) = 0;
  virtual void AddDataFromList(SEXP r_data_list) = 0;

  // Allocates and fills the appropriate data structures needed for
  // forecasting, held by the child classes.
  // Args:
  //    r_prediction_data: An R list containing data needed for
  //      prediction.
  //
  // Returns:
  //    The number of periods to be forecast.
  virtual int UnpackForecastData(SEXP r_prediction_data) = 0;

  // Unpacks forecast data for the dynamic regression state component,
  // if one is present in the model.
  // Args:
  //   model:  The model to be forecast.
  //   r_state_specification: The R list of state specfication
  //     elements, used to determine the position of the dynamic
  //     regression component.
  //   r_prediction_data: A list.  If a dynamic regression component
  //     is present this list must contain an element named
  //     "dynamic.regression.predictors", which is an R matrix
  //     containing the forecast predictors for the dynamic regression
  //     component.
  void UnpackDynamicRegressionForecastData(
      StateSpaceModelBase *model,
      SEXP r_state_specification,
      SEXP r_prediction_data);

  // This function must not be called before UnpackForecastData.  It
  // takes the current state of the model held by the child classes,
  // along with the data obtained by UnpackForecastData(), and
  // simulates one draw from the posterior predictive forecast
  // distribution.
  virtual Vector SimulateForecast(const Vector &final_state) = 0;

  virtual int UnpackHoldoutData(SEXP r_holdout_data) = 0;
  virtual Vector HoldoutDataOneStepHoldoutPredictionErrors(
      const Vector &final_state) = 0;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_MODEL_MANAGER_H_
