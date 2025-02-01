#pragma once

#include <Eigen/Dense>

namespace localise {
class Landmark {
   public:
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;
    Landmark(const Vector& position_) : position(position_) {};
    [[nodiscard]] const Vector& GetPos() const { return position; };

   protected:
    Vector position;
};
}  // namespace localise