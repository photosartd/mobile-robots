#include "ekf.h"

#include "localisation.h"

namespace localise {
using Vector = LocalisationAlgorithm::Vector;
using Matrix = LocalisationAlgorithm::Matrix;

[[nodiscard]] const std::shared_ptr<Landmark> ExtendedKalmanFilter::match(
    const Vector& z_real, const Vector& x_k_k_1, const Matrix& C_k_k_1) const {
    double min_machalonobis = std::numeric_limits<double>::max();
    std::shared_ptr<Landmark> best_match = nullptr;
    for (const auto& landmark : observation_model_->landmarks_) {
        Vector z_cap = observation_model_->z(landmark, false);
        Matrix Hk = observation_model_->GetHk(landmark, &x_k_k_1);
        Matrix Sk_ = Sk(Hk, C_k_k_1, observation_model_->GetNk(),
                      observation_model_->GetVk());
        Vector nu = z_real - z_cap;
        double dist = Machalonobis(nu, Sk_);
        if (dist < min_machalonobis) {
            min_machalonobis = dist;
            best_match = landmark;
        }
    }
    return best_match;
}

ExtendedKalmanFilter::ExtendedKalmanFilter(size_t dim) {
    x_ = Vector::Zero(dim);
    cov_ = Matrix::Zero(dim, dim);
    dim_ = dim;
}
ExtendedKalmanFilter::ExtendedKalmanFilter(const Vector& x,
                                           const Matrix& covariance) {
    x_ = x;
    cov_ = covariance;
    dim_ = static_cast<size_t>(x.rows());
}

void ExtendedKalmanFilter::update(const Vector& x_k_k_1,
                                  const Matrix& c_k_k_1) {
    Vector z_real = observation_model_->z(observation_model_->sample(), true);
    std::shared_ptr<Landmark> landmark = match(z_real, x_k_k_1, c_k_k_1);
    if (landmark == nullptr) {
        std::runtime_error("No landmark matched");
    }
    Vector z_cap = observation_model_->z(landmark, false);
    Matrix Hk = observation_model_->GetHk(landmark, &x_k_k_1);
    Matrix Sk_ = Sk(Hk, c_k_k_1, observation_model_->GetNk(),
                  observation_model_->GetVk());
    Matrix K = KalmanGain(Hk, c_k_k_1, Sk_);
    Vector x_k_k = x_k_k_1 + K * (z_real - z_cap);
    Matrix C_k_k = c_k_k_1 - K * Sk_ * K.transpose();
    x_ = x_k_k;
    cov_ = C_k_k;
    return;
}

/**
 * @brief Computes the innovation covariance matrix (Sk).
 *
 * Sk = Hk * C_k_k_1 * Hk^T + Vk * Nk * Vk^T
 *
 * @param Hk Observation matrix (m x n): Projects state covariance into
 * measurement space.
 * @param C_k_k_1 Predicted state covariance (n x n): Uncertainty in the
 * predicted state estimate.
 * @param Nk Measurement noise covariance (m x m): Represents sensor noise
 * characteristics.
 * @param Vk Noise transformation matrix (m x m): Maps noise into the
 * measurement space.
 * @return Innovation covariance matrix (m x m).
 */
Matrix ExtendedKalmanFilter::Sk(const Matrix& Hk, const Matrix& c_k_k_1,
                                const Matrix& Nk, const Matrix& Vk) {
    Matrix result = Hk * c_k_k_1 * Hk.transpose() + Vk * Nk * Vk.transpose();
    return result;
}

/**
 * @brief Computes Kalman gain
 *
 * @param Hk Observation matrix (m x n): Projects state covariance into
 * measurement space.
 * @param C_k_k_1 Predicted state covariance (n x n): Uncertainty in the
 * predicted state estimate.
 * @param Sk Covariance of innovation
 */
Matrix ExtendedKalmanFilter::KalmanGain(const Matrix& Hk, const Matrix& c_k_k_1,
                                        const Matrix& Sk) {
    Matrix result = c_k_k_1 * Hk.transpose() * Sk.inverse();
    return result;
}
}  // namespace localise