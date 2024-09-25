#include <CarConfig.h>
#include <pybind11/embed.h>
namespace py = pybind11;

void CarConfig::load_config(std::string racecar_version) {
    py::scoped_interpreter guard{}; // Start the Python interpreter

    py::module car_config_module = py::module::import("pbl_config");  // Import your Python module
    py::object load_car_config_ros = car_config_module.attr("load_car_config_ros");

    py::dict config_dict = load_car_config_ros(racecar_version);  // Call Python function and get the result
    m = config_dict["m"].cast<double>();
    Iz = config_dict["Iz"].cast<double>();
    lf = config_dict["lf"].cast<double>();
    lr = config_dict["lr"].cast<double>();
    h_cg = config_dict["h_cg"].cast<double>();
}
