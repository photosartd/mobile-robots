#include "sensor.h"

#include <random>

namespace localise {
[[nodiscard]] double LambdaSensor::h(const Sensor::Vector& x,
                                     const Sensor::Matrix& cov,
                                     const std::shared_ptr<Landmark>& landmark,
                                     double noise) const {
    double measurment = lambda_h_(x, cov, landmark);
    if (noise > 0.0) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, noise);
        measurment += distribution(generator);
    }
    return measurment;
}
}  // namespace localise