// test/test_robosimcpp_advanced.cpp

#include <gtest/gtest.h>
#include "localise.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// ----------------------------------------------------------------------
// Advanced Landmark Tests
// ----------------------------------------------------------------------
TEST(LandmarkAdvancedTest, PositionImmutability) {
    VectorXd pos(3);
    pos << 1, 2, 3;
    auto landmark = std::make_shared<localise::Landmark>(pos);
    const auto& pos_out = landmark->GetPos();
    ASSERT_EQ(pos_out.size(), pos.size());
    for (int i = 0; i < pos_out.size(); ++i) {
        EXPECT_DOUBLE_EQ(pos_out(i), pos(i));
    }
}

// ----------------------------------------------------------------------
// Advanced LambdaSensor Tests
// ----------------------------------------------------------------------
// Test that when noise is added the measured value varies across calls.
TEST(LambdaSensorAdvancedTest, NoiseVariance) {
    // Lambda returns a constant value.
    auto lambda_h = [](const VectorXd & /*x*/, const MatrixXd & /*cov*/,
                       const std::shared_ptr<localise::Landmark>& /*landmark*/) -> double {
        return 100.0;
    };
    // HRow lambda simply returns a vector of ones.
    auto lambda_HRow = [](const VectorXd &x, const MatrixXd & /*cov*/,
                          const std::shared_ptr<localise::Landmark>& /*landmark*/) -> VectorXd {
        VectorXd row(x.size());
        row.setConstant(1.0);
        return row;
    };

    auto sensor = std::make_shared<localise::LambdaSensor>(lambda_h, lambda_HRow);
    VectorXd x(1); 
    x << 0;
    MatrixXd cov = MatrixXd::Identity(1, 1);
    auto landmark = std::make_shared<localise::Landmark>(x);
    
    const int numSamples = 200;
    std::vector<double> measurements;
    for (int i = 0; i < numSamples; i++) {
        double meas = sensor->h(x, cov, landmark, 2.0); // noise sigma = 2.0
        measurements.push_back(meas);
    }
    double mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    double variance = 0.0;
    for (double m : measurements) {
        variance += (m - mean) * (m - mean);
    }
    variance /= measurements.size();
    // With sigma=2.0 the variance should be roughly 4. Allow some tolerance.
    EXPECT_NEAR(variance, 4.0, 2.0);
}

// ----------------------------------------------------------------------
// Advanced ObservationModel Tests
// ----------------------------------------------------------------------
// Test that GetHk returns the derivative row computed by the sensor.
// Here we simulate a sensor whose HRow equals the difference between state and landmark.
TEST(ObservationModelAdvancedTest, GetHk) {
    auto lambda_h = [](const VectorXd &x, const MatrixXd & /*cov*/,
                       const std::shared_ptr<localise::Landmark>& landmark) -> double {
        return (x - landmark->GetPos()).norm();
    };
    auto lambda_HRow = [](const VectorXd &x, const MatrixXd & /*cov*/,
                          const std::shared_ptr<localise::Landmark>& landmark) -> VectorXd {
        return (x - landmark->GetPos());
    };
    auto sensor = std::make_shared<localise::LambdaSensor>(lambda_h, lambda_HRow);
    
    localise::ConstantNoiseObservationModel obs(0.5);
    obs.AddSensor(sensor);
    VectorXd state(2);
    state << 5, 5;
    MatrixXd cov = MatrixXd::Identity(2, 2);
    obs.SetState(state, cov);
    auto landmark = std::make_shared<localise::Landmark>(VectorXd::Zero(2));
    std::vector<std::shared_ptr<localise::Landmark>> landmarks = {landmark};
    obs.SetLandmarks(landmarks);
    
    MatrixXd Hk = obs.GetHk(landmark, &state);
    ASSERT_EQ(Hk.rows(), 1);
    ASSERT_EQ(Hk.cols(), 2);
    EXPECT_NEAR(Hk(0, 0), 5.0, 1e-5);
    EXPECT_NEAR(Hk(0, 1), 5.0, 1e-5);
}

// Test that the sampling method of the observation model selects from all landmarks.
TEST(ObservationModelAdvancedTest, SamplingDistribution) {
    VectorXd pos1(2); pos1 << 1, 1;
    VectorXd pos2(2); pos2 << 2, 2;
    VectorXd pos3(2); pos3 << 3, 3;
    auto lm1 = std::make_shared<localise::Landmark>(pos1);
    auto lm2 = std::make_shared<localise::Landmark>(pos2);
    auto lm3 = std::make_shared<localise::Landmark>(pos3);
    
    localise::ConstantNoiseObservationModel obs(1.0);
    std::vector<std::shared_ptr<localise::Landmark>> landmarks = {lm1, lm2, lm3};
    obs.SetLandmarks(landmarks);
    
    int count1 = 0, count2 = 0, count3 = 0;
    const int numSamples = 1000;
    for (int i = 0; i < numSamples; ++i) {
        auto sample = obs.sample();
        if (sample == lm1)
            count1++;
        else if (sample == lm2)
            count2++;
        else if (sample == lm3)
            count3++;
    }
    EXPECT_GT(count1, 0);
    EXPECT_GT(count2, 0);
    EXPECT_GT(count3, 0);
}

// ----------------------------------------------------------------------
// Advanced ExtendedKalmanFilter Static Function Tests
// ----------------------------------------------------------------------
// Test that Sk and KalmanGain compute the expected values given sample matrices.
TEST(ExtendedKalmanFilterStaticTest, SkAndKalmanGain) {
    MatrixXd Hk(1, 2);
    Hk << 1, 2;
    MatrixXd C(2, 2);
    C << 2, 0, 0, 2;
    MatrixXd Nk(1, 1);
    Nk << 1;
    MatrixXd Vk(1, 1);
    Vk << 1;
    
    MatrixXd expectedSk = Hk * C * Hk.transpose() + Vk * Nk * Vk.transpose();
    MatrixXd computedSk = localise::ExtendedKalmanFilter::Sk(Hk, C, Nk, Vk);
    ASSERT_EQ(computedSk.rows(), expectedSk.rows());
    ASSERT_EQ(computedSk.cols(), expectedSk.cols());
    for (int i = 0; i < computedSk.rows(); i++) {
        for (int j = 0; j < computedSk.cols(); j++) {
            EXPECT_NEAR(computedSk(i, j), expectedSk(i, j), 1e-5);
        }
    }
    
    MatrixXd computedK = localise::ExtendedKalmanFilter::KalmanGain(Hk, C, computedSk);
    MatrixXd expectedK = C * Hk.transpose() * computedSk.inverse();
    ASSERT_EQ(computedK.rows(), expectedK.rows());
    ASSERT_EQ(computedK.cols(), expectedK.cols());
    for (int i = 0; i < computedK.rows(); i++) {
        for (int j = 0; j < computedK.cols(); j++) {
            EXPECT_NEAR(computedK(i, j), expectedK(i, j), 1e-5);
        }
    }
}

// ----------------------------------------------------------------------
// Advanced ExtendedKalmanFilter Update Tests
// ----------------------------------------------------------------------
// Test that multiple updates cause the filter state to converge toward the "true" state.
TEST(ExtendedKalmanFilterAdvancedTest, UpdateConvergence) {
    // Define a sensor that "measures" the first element of the state.
    auto lambda_h = [](const VectorXd &x, const MatrixXd & /*cov*/,
                       const std::shared_ptr<localise::Landmark>& /*landmark*/) -> double {
        return x(0);
    };
    // Its Jacobian row is [1, 0, 0, ...].
    auto lambda_HRow = [](const VectorXd &x, const MatrixXd & /*cov*/,
                          const std::shared_ptr<localise::Landmark>& /*landmark*/) -> VectorXd {
        VectorXd row(x.size());
        row.setZero();
        row(0) = 1.0;
        return row;
    };
    auto sensor = std::make_shared<localise::LambdaSensor>(lambda_h, lambda_HRow);
    
    auto obs_model = std::make_shared<localise::ConstantNoiseObservationModel>(0.1);
    obs_model->AddSensor(sensor);
    VectorXd true_state(2);
    true_state << 10, 5;
    MatrixXd true_cov = MatrixXd::Identity(2, 2);
    obs_model->SetState(true_state, true_cov);
    
    auto landmark = std::make_shared<localise::Landmark>(true_state);
    std::vector<std::shared_ptr<localise::Landmark>> landmarks = {landmark};
    obs_model->SetLandmarks(landmarks);
    
    // Start with an initial state far from the true state.
    VectorXd init_state(2);
    init_state << 0, 0;
    MatrixXd init_cov = MatrixXd::Identity(2, 2) * 10;
    localise::ExtendedKalmanFilter ekf(init_state, init_cov);
    ekf.SetObservationModel(obs_model);
    
    // Run several updates.
    for (int i = 0; i < 20; i++){
        ekf.update(ekf.GetState(), ekf.GetCovariance());
    }
    VectorXd final_state = ekf.GetState();
    // Expect convergence toward the true state's first element.
    EXPECT_NEAR(final_state(0), 10, 1.0);
}

// ----------------------------------------------------------------------
// Advanced ExtendedKalmanFilter Matching Tests
// ----------------------------------------------------------------------
// Test that the match() method selects the landmark closest to the state.
TEST(ExtendedKalmanFilterAdvancedTest, MatchReturnsClosestLandmark) {
    VectorXd pos1(2); pos1 << 0, 0;
    VectorXd pos2(2); pos2 << 10, 10;
    auto lm1 = std::make_shared<localise::Landmark>(pos1);
    auto lm2 = std::make_shared<localise::Landmark>(pos2);
    
    // Sensor that returns the Euclidean distance between state and landmark.
    auto lambda_h = [](const VectorXd &x, const MatrixXd & /*cov*/,
                       const std::shared_ptr<localise::Landmark>& landmark) -> double {
        return (x - landmark->GetPos()).norm();
    };
    // Jacobian row: a unit vector in the direction from landmark to state.
    auto lambda_HRow = [](const VectorXd &x, const MatrixXd & /*cov*/,
                          const std::shared_ptr<localise::Landmark>& landmark) -> VectorXd {
        VectorXd diff = x - landmark->GetPos();
        if(diff.norm() > 0)
            return diff / diff.norm();
        else
            return VectorXd::Zero(x.size());
    };
    auto sensor = std::make_shared<localise::LambdaSensor>(lambda_h, lambda_HRow);
    
    auto obs_model = std::make_shared<localise::ConstantNoiseObservationModel>(0.1);
    obs_model->AddSensor(sensor);
    VectorXd state(2);
    state << 1, 1;
    MatrixXd cov = MatrixXd::Identity(2, 2) * 0.5;
    obs_model->SetState(state, cov);
    
    std::vector<std::shared_ptr<localise::Landmark>> landmarks = {lm1, lm2};
    obs_model->SetLandmarks(landmarks);
    
    localise::ExtendedKalmanFilter ekf(state, cov);
    ekf.SetObservationModel(obs_model);
    
    auto measurement = obs_model->z(lm1, false);
    auto matched = ekf.match(measurement, state, cov);
    // Since the state (1,1) is closer to (0,0) than (10,10), we expect lm1.
    EXPECT_EQ(matched, lm1);
}
