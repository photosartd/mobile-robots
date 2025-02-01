#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

#include "landmark.h"

namespace localise {
class Sensor {
   public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    virtual ~Sensor() = default;

    /**
     * @brief Measurement: Represents the sensor reading. TODO: may need to take
     * a vector of Landmarks
     */
    [[nodiscard]] virtual double h(const Vector& x, const Matrix& cov,
                                   const std::shared_ptr<Landmark>& landmark,
                                   double noise = 0.0) const = 0;

    /**
     * @brief Jacobian row: Returns the partial derivative of the measurement
     * with respect to the landmark positions at the given x, cov and landmark
     * position.
     */
    [[nodiscard]] virtual Vector HRow(
        const Vector& x, const Matrix& cov,
        const std::shared_ptr<Landmark>& landmark) const = 0;
};

class LambdaSensor : public Sensor {
   public:
    template <typename T>
    using Lambda = std::function<T(const Vector&, const Matrix&,
                                   const std::shared_ptr<Landmark>&)>;

    explicit LambdaSensor(Lambda<double> lambda_h, Lambda<Vector> lambda_HRow)
        : lambda_h_(std::move(lambda_h)),
          lambda_HRow_(std::move(lambda_HRow)) {}

    [[nodiscard]] double h(const Vector& x, const Matrix& cov,
                           const std::shared_ptr<Landmark>& landmark,
                           double noise) const override;

    [[nodiscard]] Vector HRow(
        const Vector& x, const Matrix& cov,
        const std::shared_ptr<Landmark>& landmark) const override {
        return lambda_HRow_(x, cov, landmark);
    }

   private:
    Lambda<double> lambda_h_;
    Lambda<Vector> lambda_HRow_;
};
}  // namespace localise