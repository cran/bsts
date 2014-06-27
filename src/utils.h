// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#ifndef BSTS_SRC_UTILS_H_
#define BSTS_SRC_UTILS_H_

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace bsts {

// Factory method for creating a StateModel based on inputs supplied
// to R.  Returns a BOOM-style smart pointer to the StateModel that
// gets created.
// Args:
//   list_arg: The portion of the state.specification list (that was
//     supplied to R by the user), corresponding to the state model
//     that needs to be created
//   io_manager: A pointer to the object manaaging the R list that
//     will record (or has already recorded) the MCMC output
// Returns:
//   A BOOM smart pointer to a StateModel that can be added to a
//   StateSpaceModelBase.
BOOM::Ptr<BOOM::StateModel> CreateStateModel(
    SEXP list_arg, BOOM::RListIoManager *io_manager);

//======================================================================
// Record the state of a DynamicRegressionStateModel in the io_manager.
// Args:
//   model: The state space model that may or may not have a dynamic
//     regression state component to be recorded.
//   io_manager: The io_manager in charge of building the list
//     containing the dynamic regression coefficients.
void RecordDynamicRegression(BOOM::StateSpaceModelBase * model,
                             BOOM::RListIoManager *io_manager);

// A callback class for saving one step ahead prediction errors from
// the Kalman filter.
class PredictionErrorCallback : public BOOM::VectorIoCallback {
 public:
  explicit PredictionErrorCallback(BOOM::StateSpaceModelBase *model)
      : model_(model) {}

  // Each element is a vector of one step ahead prediction errors, so
  // the dimension is the time dimension of the model.
  virtual int dim()const {
    return model_->time_dimension();
  }

  virtual BOOM::Vec get_vector()const {
    return model_->one_step_prediction_errors();
  }

 private:
  BOOM::StateSpaceModelBase *model_;
};

}  // namespace bsts
#endif  // BSTS_SRC_UTILS_H_
