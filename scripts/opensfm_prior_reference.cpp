// Validation only: invoke the actual unmodified OpenSfM template functors.
// Compile against the pinned reference checkout; not part of runtime execution.
#include <bundle/error/absolute_motion_errors.h>
#include <bundle/error/prior_error.h>
#include <ceres/loss_function.h>

struct PositionData : bundle::Data<Vec3d> {
  PositionData(const Vec3d &v,const Vec3d &p,const Vec3d &s):Data("camera",v,p,s){Init();}
  void ValueToData(const Vec3d &v,VecXd &d) const override {d=v;}
  void DataToValue(const VecXd &d,Vec3d &v) const override {v=d;}
};

extern "C" void opensfm_position(const double *center,const double *prior,const double *sigma,double *r) {
  PositionData data{Eigen::Map<const Vec3d>(center),Eigen::Map<const Vec3d>(prior),Eigen::Map<const Vec3d>(sigma)};
  bundle::DataPriorError<Vec3d> functor(&data);
  const double *params[]={center};
  // indexes_ is an unordered_set, so match the actual native residual order
  // to canonical x,y,z using distinct test data in the Python validator.
  functor(params,r);
}
extern "C" void opensfm_up(const double *rotation,const double *up,double sigma,double *r,double *cost) {
  double pose[6]={rotation[0],rotation[1],rotation[2],0,0,0},identity[6]={0,0,0,0,0,0};
  bundle::UpVectorError functor(Eigen::Map<const Vec3d>(up),sigma);
  functor(pose,identity,r);
  double rho[3];ceres::CauchyLoss loss(1);loss.Evaluate(r[0]*r[0]+r[1]*r[1]+r[2]*r[2],rho);
  cost[0]=.5*rho[0];cost[1]=rho[1];
}
