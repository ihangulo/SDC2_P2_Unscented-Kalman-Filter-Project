#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.36;  // get this value from Trial & error

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.50;  // get this value from Trial & error

  /**
  In the starter code, we have given values for the process noise and measurement noise.
  You will need to tune the process noise parameters std_a_ and std_yawdd_
  in order to get your solution working on both datasets.
  The measurement noise parameters for lidar and radar should be left as given.
  */
   // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;
  previous_timestamp_ = 0;

  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

  // set radar measurement dimension
  n_z_= 3 ;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //  NIS for radar
  NIS_radar_ = 0.0;

  //  NIS for laser
  NIS_laser_ = 0.0;


  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 0.3, 0, 0, 0, 0,
        0,0.3, 0, 0, 0,
        0, 0, 0.3, 0, 0,
        0, 0, 0, 0.3, 0,
        0, 0, 0, 0, 0.3;

  // Laser

  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_, 0,
            0, std_laspy_;

  //set vector for weights (Sigma points)
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR ) {

     /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = std::max(0.001, meas_package.raw_measurements_(0));
      float phi = std::max(0.001, meas_package.raw_measurements_(1));
      float rhodot = std::max(0.001, meas_package.raw_measurements_(2));

      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx =  rhodot * cos(phi);
      float vy = rhodot * sin(phi);

      x_ << px,py,sqrt(vx*vx +vy*vy),0.0,0.0;

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER ) {

      /**
        Initialize state.
      */
      float px = meas_package.raw_measurements_[0];
	    float py = meas_package.raw_measurements_[1];

      if (fabs(px) <0.01 && fabs(py) <0.01) { // px == 0 -->Nan Error occured
        px = 0.01;
        py = 0.01;
      }
      //set the state with the initial location and zero velocity
      x_ << px, py, 0.0, 0.0, 0.0;

    }
    previous_timestamp_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;

  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && ! use_radar_ )
    return;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  // compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0f;	//dt - expressed in seconds

  Prediction(dt);

  previous_timestamp_ = meas_package.timestamp_;

  /*****************************************************************************
   *  Update
  ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR  && use_radar_ ) {

    // mean predicted measurement
    VectorXd z_pred ;// will be allocated on PredictRadarMeasurement = VectorXd::Zero(n_z_);

    // measurement covariance matrix
    MatrixXd S ;// will be allocated on PredictRadarMeasurement = MatrixXd::Zero(n_z_,n_z_);

    // sigma points in measurement space
    MatrixXd Zsig ;// will be allocated on PredictRadarMeasurement = MatrixXd(n_z_, 2 * n_aug_ + 1);

    // get z_pred, S, Zsig
    PredictRadarMeasurement(z_pred,S, Zsig);

    // update x_, P_, NIS_radar_
    UpdateRadar(meas_package, z_pred, S, Zsig);


  } else if( meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_ ) {
   // Update Lidar
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //create sigma point matrix
  MatrixXd Xsig_aug; // will be allocated on AugmentedSigmaPoints() // = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // predicted sigma points matrix
  MatrixXd Xsig_pred; // will be allocated on SigmaPointPrediction //=  MatrixXd(n_x_, 2 * n_x_ + 1);

  AugmentedSigmaPoints(Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t, Xsig_pred);

  VectorXd x_pred; // will be allocated on PredictMeanAndCovariance() = VectorXd(n_x_);
  MatrixXd P_pred ; //will be allocated on PredictMeanAndCovariance() = MatrixXd(n_x_, n_x_);
  PredictMeanAndCovariance(Xsig_pred ,x_pred, P_pred);

  x_ = x_pred;
  P_ = P_pred;
  Xsig_pred_ = Xsig_pred;

}

/**
 * Get argmented sigma points
 */
void UKF::AugmentedSigmaPoints(MatrixXd &Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;

  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //write result
  Xsig_out = Xsig_aug;

}


void UKF::SigmaPointPrediction(const MatrixXd &Xsig_aug, const double delta_t, MatrixXd &Xsig_out) {

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;
    double yawd_delta_t= yawd*delta_t;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd_delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd_delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd_delta_t;
    double yawd_p = yawd;

    double deltat_daltat = delta_t*delta_t;
    double nu_a_deltat_daltat = nu_a*deltat_daltat;

    //add noise
    px_p = px_p + 0.5*nu_a_deltat_daltat* cos(yaw);
    py_p = py_p + 0.5*nu_a_deltat_daltat * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*deltat_daltat;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //write result
  Xsig_out = Xsig_pred;

  return;

}


void UKF::PredictMeanAndCovariance(const MatrixXd &Xsig_pred, VectorXd &x_out, MatrixXd &P_out) {

  //create vector for weights
  //VectorXd weights = VectorXd(2*n_aug_+1);

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // simple calculation
  double pi_2 = 2.*M_PI;


  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;

    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=pi_2;
    while (x_diff(3)<-M_PI) x_diff(3)+=pi_2;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //write result
  x_out = x;
  P_out = P;
}


void UKF::PredictRadarMeasurement(VectorXd &z_out, MatrixXd &S_out, MatrixXd &Zsig_out) {

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;


    // Avoid division by zero
    if(fabs(p_x) <= 0.0001){
      p_x = 0.0001;
    }
    if(fabs(p_y) <= 0.0001){
      p_y = 0.0001;
    }

    // measurement model

    double px_px = p_x*p_x;
    double py_py = p_y*p_y;
    double sqrt_px2_py2 = sqrt(px_px + py_py);


    Zsig(0,i) = sqrt_px2_py2;                       //r
    Zsig(1,i) = atan2(p_y,p_x);                            //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt_px2_py2;  //r_dot
  }

  // simple calculation
  double pi_2 = 2.*M_PI;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=pi_2;
    while (z_diff(1)<-M_PI) z_diff(1)+=pi_2;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix

  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  z_out = z_pred;
  S_out = S;
  Zsig_out = Zsig;

}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  VectorXd z = meas_package.raw_measurements_;

  VectorXd z_pred = H_laser_ * x_;

  VectorXd y = z - z_pred;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_laser_ * PHt + R_laser_; // H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();

  MatrixXd K = PHt * Si;
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_laser_) * P_;

  NIS_laser_  = y.transpose() * Si * y;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage meas_package, const VectorXd &z_pred, const MatrixXd &S, const MatrixXd &Zsig) {

  VectorXd z = VectorXd::Zero(n_z_);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //print result
  //std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  //std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}




