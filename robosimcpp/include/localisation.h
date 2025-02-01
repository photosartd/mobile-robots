#pragma once
#include <Eigen/Dense>
#include <memory>

#include "observation_model.h"
#include "landmark.h"

namespace localise {
class LocalisationAlgorithm {
   public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

   public:
    virtual void update(const Vector& x_k_k_1, const Matrix& c_k_k_1) = 0;
    [[nodiscard]] const Vector& GetState() const { return x_; };
    [[nodiscard]] const Matrix& GetCovariance() const { return cov_; };
    virtual ~LocalisationAlgorithm() = default;
    void SetObservationModel(std::shared_ptr<ObservationModel> observation_model) {
        observation_model_ = std::move(observation_model);
    }

    static double Machalonobis(const Vector& delta, const Matrix& cov) {
        return delta.transpose() * cov.inverse() * delta;
    };

   protected:
    Vector x_;
    Matrix cov_;
    size_t dim_;

    std::shared_ptr<ObservationModel> observation_model_;
};
}  // namespace localise
