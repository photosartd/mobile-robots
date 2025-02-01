#pragma once
#include "localisation.h"
#include "landmark.h"

namespace localise {
class ExtendedKalmanFilter : public LocalisationAlgorithm {
   public:
    ExtendedKalmanFilter(size_t dim);
    ExtendedKalmanFilter(const Vector& x, const Matrix& covariance);
    virtual void update(const Vector& x_k_k_1, const Matrix& c_k_k_1);
    virtual ~ExtendedKalmanFilter() = default;
    // EKF related methods
    static Matrix Sk(const Matrix& Hk, const Matrix& c_k_k_1, const Matrix& Nk,
                     const Matrix& Vk);
    static Matrix KalmanGain(const Matrix& Hk, const Matrix& c_k_k_1,
                             const Matrix& Sk);
    [[nodiscard]] const std::shared_ptr<Landmark> match(
        const Vector& z_real, const Vector& x_k_k_1,
        const Matrix& C_k_k_1) const;
};
}  // namespace localise