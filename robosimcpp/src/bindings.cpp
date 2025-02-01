#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <exception>
#include <iostream>
#include <memory>

#include "localise.h"  // aggregator header

namespace py = pybind11;
using namespace localise;

PYBIND11_MODULE(pylocalise, m) {
    try {
        m.doc() = "Pybind11 bindings for localise classes";

        // 1) Landmark
        py::class_<Landmark, std::shared_ptr<Landmark>>(m, "Landmark",
                                                        py::module_local())
            .def(py::init<const Eigen::VectorXd &>())
            .def("GetPos", &Landmark::GetPos);

        // 2) Sensor (abstract)
        py::class_<Sensor, std::shared_ptr<Sensor>>(m, "Sensor");
        // 3) LambdaSensor (concrete)
        py::class_<LambdaSensor, Sensor, std::shared_ptr<LambdaSensor>>(
            m, "LambdaSensor")
            .def(py::init<std::function<double(
                              const Eigen::VectorXd &, const Eigen::MatrixXd &,
                              const std::shared_ptr<Landmark> &)>,
                          std::function<Eigen::VectorXd(
                              const Eigen::VectorXd &, const Eigen::MatrixXd &,
                              const std::shared_ptr<Landmark> &)>>(),
                 py::arg("lambda_h"), py::arg("lambda_HRow"))
            .def("h", &LambdaSensor::h, py::arg("x"), py::arg("cov"),
                 py::arg("landmark"), py::arg("noise") = 0.0)
            .def("HRow", &LambdaSensor::HRow);

        // 4) ObservationModel (abstract)
        py::class_<ObservationModel, PyObservationModel,
                   std::shared_ptr<ObservationModel>>(m, "ObservationModel")
            .def(py::init<>())
            .def("AddSensor", &ObservationModel::AddSensor)
            .def("SetLandmarks", &ObservationModel::SetLandmarks)
            .def("SetState", &ObservationModel::SetState)
            .def("sample", &ObservationModel::sample)
            .def("GetState", &ObservationModel::GetState)
            .def("GetCovariance", &ObservationModel::GetCovariance)
            .def("z", &ObservationModel::z, py::arg("sample"),
                 py::arg("noise") = false)
            .def("GetVk", &ObservationModel::GetVk)
            .def("GetNk", &ObservationModel::GetNk)
            .def(
                "GetHk",
                [](const ObservationModel &self,
                const std::shared_ptr<Landmark> &sample,
                const Eigen::VectorXd *x) { return self.GetHk(sample, x); },
                py::arg("sample"), py::arg("x") = nullptr);

        // 5) ConstantNoiseObservationModel (concrete)
        py::class_<ConstantNoiseObservationModel, ObservationModel,
                   std::shared_ptr<ConstantNoiseObservationModel>>(
            m, "ConstantNoiseObservationModel")
            .def(py::init<double>(), py::arg("sigma"))
            .def("z", &ConstantNoiseObservationModel::z, py::arg("sample"),
                 py::arg("noise") = false);

        // 6) LocalisationAlgorithm (abstract)
        py::class_<LocalisationAlgorithm,
                   std::shared_ptr<LocalisationAlgorithm>>(
            m, "LocalisationAlgorithm")
            .def("update", &LocalisationAlgorithm::update)
            .def_static("Machalonobis", &LocalisationAlgorithm::Machalonobis)
            .def("GetState", &LocalisationAlgorithm::GetState)
            .def("GetCovariance", &LocalisationAlgorithm::GetCovariance)
            .def("SetObservationModel",
                 &LocalisationAlgorithm::SetObservationModel);

        // 7) ExtendedKalmanFilter (concrete)
        py::class_<ExtendedKalmanFilter, LocalisationAlgorithm,
                   std::shared_ptr<ExtendedKalmanFilter>>(
            m, "ExtendedKalmanFilter")
            .def(py::init<size_t>())
            .def(py::init<const Eigen::VectorXd&, const Eigen::MatrixXd&>())
            .def("update", &ExtendedKalmanFilter::update)
            .def("match", &ExtendedKalmanFilter::match)
            .def_static("Sk", &ExtendedKalmanFilter::Sk)
            .def_static("KalmanGain", &ExtendedKalmanFilter::KalmanGain);
    } catch (const std::exception &e) {
        // 2) Print/log something to stderr:
        std::cerr << "[pylocalise] Caught exception in module init: "
                  << e.what() << std::endl;

        // 3) Also report it to Python by setting a Python exception:
        PyErr_SetString(PyExc_RuntimeError, e.what());

        throw pybind11::error_already_set();
    }
}