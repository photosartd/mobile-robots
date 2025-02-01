#pragma once

#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "landmark.h"
#include "sensor.h"

namespace localise {
class ObservationModel {
   public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    virtual ~ObservationModel() = default;

   protected:
    const Vector* x_;
    const Matrix* cov_;
    std::vector<std::shared_ptr<Sensor>> sensors_;

   public:
    std::vector<std::shared_ptr<Landmark>> landmarks_;

   public:
    void AddSensor(std::shared_ptr<Sensor> sensor) {
        sensors_.push_back(std::move(sensor));
    }

    void SetLandmarks(const std::vector<std::shared_ptr<Landmark>>& landmarks) {
        landmarks_.clear();
        landmarks_.reserve(landmarks.size());
        for (const auto& landmark : landmarks) {
            landmarks_.push_back(landmark);  // Shared ownership
        }
    }

    void SetState(const Vector& x, const Matrix& cov) {
        x_ = &x;
        cov_ = &cov;
    }

    [[nodiscard]] const std::shared_ptr<Landmark> sample() const {
        if (landmarks_.empty()) {
            return nullptr;  // No landmarks available
        }

        // Initialize random engine
        static std::random_device rd;
        static std::mt19937 generator(rd());

        // Create a uniform distribution over landmark indices
        std::uniform_int_distribution<size_t> distribution(
            0, landmarks_.size() - 1);

        // Select a random landmark
        size_t random_index = distribution(generator);
        return landmarks_[random_index];
    }

    [[nodiscard]] const Vector& GetState() const { return *x_; }
    [[nodiscard]] const Matrix& GetCovariance() const { return *cov_; }

    // Sensors
    [[nodiscard]] virtual Vector z(const std::shared_ptr<Landmark>& sample,
                                   bool noise) const {
        Vector result(sensors_.size());
        std::transform(sensors_.begin(), sensors_.end(), result.data(),
                       [&](const std::shared_ptr<Sensor>& sensor) {
                           return sensor->h(*x_, *cov_, sample);
                       });
        return result;
    };
    /**
     * @brief Observation matrix (m x n): Projects state covariance into
     * measurement space.
     */
    [[nodiscard]] virtual Matrix GetHk(const std::shared_ptr<Landmark>& sample,
                                       const Vector* x = nullptr) const {
        Matrix result(sensors_.size(), x_->size());
        for (size_t i = 0; i < sensors_.size(); i++) {
            auto x_curr =
                x ? *x
                  : *x_;  // To be able to provide a position after prediction
            result.row(i) = sensors_[i]->HRow(x_curr, *cov_, sample);
        }
        return result;
    };
    /**
     * @brief Measurement noise covariance (m x m): Represents sensor noise
     * characteristics.
     */
    [[nodiscard]] virtual Matrix GetNk() const = 0;
    /**
     * @brief Noise transformation matrix (m x m): Maps noise into the
     * measurement space.
     */
    [[nodiscard]] virtual Matrix GetVk() const {
        return Matrix::Identity(sensors_.size(), sensors_.size());
    };
};

// A trampoline class for ObservationModel
class PyObservationModel : ObservationModel {
   public:
    using ObservationModel::ObservationModel;  // Inherit constructors

    Matrix GetNk() const override {
        PYBIND11_OVERRIDE_PURE(
            Matrix,            // Return type
            ObservationModel,  // Parent class
            GetNk,             // Name of function in C++ (must match exactly)
        );
    }
};

class ConstantNoiseObservationModel : public ObservationModel {
   protected:
    double noise_sigma;

   public:
    ConstantNoiseObservationModel(double sigma) : noise_sigma(sigma) {};

    [[nodiscard]] Matrix GetNk() const override {
        return noise_sigma * Matrix::Identity(sensors_.size(), sensors_.size());
    };

    [[nodiscard]] Vector z(const std::shared_ptr<Landmark>& sample,
                           bool noise) const override {
        std::vector<double> measurments;
        measurments.reserve(sensors_.size());
        for (const auto& sensor : sensors_) {
            measurments.emplace_back(
                sensor->h(*x_, *cov_, sample, noise ? noise_sigma : 0.0));
        }
        return Vector::Map(measurments.data(), measurments.size());
    }
};
}  // namespace localise