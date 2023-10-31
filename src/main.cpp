#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> //for automatic type conversion of Eigen to/from python
#include <pybind11/stl.h> // for automatic type conversion of STL containers to/from python

#include "LibProjectiveGeometry/ProjectionMatrix.h"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediateCPU.hxx"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediate.h"
#include "LibEpipolarConsistency/EpipolarConsistency.h"
#include "NRRD/nrrd_image.hxx"
#include "NRRD/nrrd_image_view.hxx"
#include "LibEpipolarConsistency/RadonIntermediate.h"

namespace py = pybind11;

PYBIND11_MODULE(ecc, m)
{
    m.doc() = "Python wrappers for epipolar consistency computations from https://github.com/aaichert/EpipolarConsistency";

    // contents of ProjectionMatrix.h
    m.def("makeCalibrationMatrix", &Geometry::makeCalibrationMatrix, "Creates a new intrinsics matrix K.",
        py::arg("ax"), py::arg("ay"), py::arg("u0"), py::arg("v0"), py::arg("skew")=0.0);
    m.def("makeProjectionMatrix", &Geometry::makeProjectionMatrix, "Creates a projection matrix from K, R and t.",
        py::arg("K"), py::arg("R"), py::arg("t"));
    m.def("computeFundamentalMatrix", &Geometry::computeFundamentalMatrix, "Computes fundamental matrix from two projection matrices.", 
        py::arg("P0"), py::arg("P1"));
    m.def("projectionMatrixDecomposition", &Geometry::projectionMatrixDecomposition, 
        "Decompose Projection Matrix into K[R|t] using RQ-decomposition. Returns false if R is left-handed (For RHS world coordinate systems, this implies imageVPointsUp is wrong).",
        py::arg("P"), py::arg("K"), py::arg("R"), py::arg("t"), py::arg("imageVPointsUp")=true);
    m.def("pseudoInverse", &Geometry::pseudoInverse, "Backprojection", py::arg("P"));
    m.def("getCameraIntrinsics", &Geometry::getCameraIntrinsics, "Return intrinsic parameters in upper triangular 3x3 matrix.",
        py::arg("P"));
    m.def("getCameraPrincipalPoint", &Geometry::getCameraPrincipalPoint, "Compute the principal point via M*m3.",
        py::arg("P"));
    m.def("getCameraDirection", &Geometry::getCameraDirection, "Direction of principal ray from a projection matrix. (Normal to principal plane, which is last row of P).",
        py::arg("P"));
    m.def("getCameraAxisDirections", &Geometry::getCameraAxisDirections, "Compute the two three-points where the image u- and v-axes meet infinity.",
        py::arg("P"));
    m.def("getCameraFocalLengthPx", &Geometry::getCameraFocalLengthPx, "Compute the focal length in pixels (diagonal entries of K in P=K[R t] ). Assumes normalized projection matrix.",
        py::arg("P"));
    m.def("getCameraImagePlane", &Geometry::getCameraImagePlane, "Decomposes the projection matrix to compute the equation of the image plane. For left-handed coordinate systems, pixel_spacing can be negated.",
        py::arg("P"), py::arg("pixel_spacing"));

    // base classes need to be wrapped, too. Otherwise python will complain about an unknown base class when importing the module.
    py::class_<NRRD::ImageView<float>>(m, "ImageViewFloat");

    // templates need to be set to a specific data type for wrapping but can be wrapped multiple times with different data types
    // pointers to data larger than one value in C++ need to be wrapped using py::buffer
    py::class_<NRRD::Image<float>, NRRD::ImageView<float>>(m, "ImageFloat2D", py::buffer_protocol())
        .def(py::init([](py::buffer const b){

            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<float>::format())
                throw std::runtime_error("Buffer must be float32.");
            if (info.ndim != 2)
                throw std::runtime_error("Buffer must have 2 dimensions.");

            //create 2D uninitialized image using other constructor
            auto v = new NRRD::Image<float>(info.shape[1], info.shape[0]);
            //copy data from buffer into new Image object
            memcpy(v->get_data(), info.ptr, sizeof(float) * (size_t) (v->size(0) * v->size(1)));
            return v;
        }), "Create a 2D image from a python buffer.", py::arg("b"))
        /// Provide buffer access
       .def_buffer([](NRRD::Image<float> &m) -> py::buffer_info {
            return py::buffer_info(
                m.get_data(),                               /* Pointer to buffer */
                { m.size(1), m.size(0) },                 /* Buffer dimensions */
                { sizeof(float) * size_t(m.size(0)),     /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    // RadonIntermediate
    py::class_<EpipolarConsistency::RadonIntermediate> radon_intermediate(m, "RadonIntermediate", py::buffer_protocol());

    radon_intermediate.def_buffer([](EpipolarConsistency::RadonIntermediate &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(float),                          /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            { m.getRadonBinNumber(1), m.getRadonBinNumber(0) },                 /* Buffer dimensions */
            { sizeof(float) * m.getRadonBinNumber(0),             /* Strides (in bytes) for each index */
              sizeof(float) }
        );
    });
    radon_intermediate.def(py::init<const NRRD::ImageView<float>&, int, int, EpipolarConsistency::RadonIntermediate::Filter, EpipolarConsistency::RadonIntermediate::PostProcess>(), 
        "Creates a RadonIntermediateObject by taking a projection image, computing the Radon transform and deriving it.", py::arg("projection_image"), py::arg("size_alpha"), py::arg("size_t"), py::arg("filter"), py::arg("post_process"));
    radon_intermediate.def("replaceRadonIntermediateData", &EpipolarConsistency::RadonIntermediate::replaceRadonIntermediateData, "Update Radon intermediate data with CPU memory.",
        py::arg("radon_intermediate_image"));
    radon_intermediate.def("getFilter", &EpipolarConsistency::RadonIntermediate::getFilter, "Which filter has been applied to the Radon transform t-direction?");
    radon_intermediate.def("isDerivative", &EpipolarConsistency::RadonIntermediate::isDerivative, "If true, then the Radon intermediate function is odd, meaning dtr(alpha+Pi,t)=-dtr(alpha,-t). If false, we have dtr(alpha+Pi,t)=dtr(alpha,-t)");
    radon_intermediate.def("readback", &EpipolarConsistency::RadonIntermediate::readback, "Readback Radon Intermediate data to CPU. If gpu_memory_only is set, the texture will be transferred to global GPU memory but not read back to RAM.", 
        py::arg("gpu_memory_only")=false);
    radon_intermediate.def("getRadonBinNumber", &EpipolarConsistency::RadonIntermediate::getRadonBinNumber, "Access to the size of the Radon transform. 0:distance 1:angle", 
        py::arg("dim"));
    radon_intermediate.def("getOriginalImageSize", &EpipolarConsistency::RadonIntermediate::getOriginalImageSize, "Access to size of original image 0:width 1:height",
        py::arg("dim"));
    radon_intermediate.def("getRadonBinSize", &EpipolarConsistency::RadonIntermediate::getRadonBinSize, "Access to spacing of DTR. 0:angle 1:distance",
        py::arg("dim")=1);
    radon_intermediate.def("data", py::overload_cast<>(&EpipolarConsistency::RadonIntermediate::data), "Access to raw data on CPU (may return invalid image). See also: readback(...)");

    py::enum_<EpipolarConsistency::RadonIntermediate::Filter>(radon_intermediate, "Filter")
        .value("Derivative", EpipolarConsistency::RadonIntermediate::Filter::Derivative)
        .value("Ramp", EpipolarConsistency::RadonIntermediate::Filter::Ramp)
        .value("None", EpipolarConsistency::RadonIntermediate::Filter::None)
        .export_values();

    py::enum_<EpipolarConsistency::RadonIntermediate::PostProcess>(radon_intermediate, "PostProcess")
        .value("Identity", EpipolarConsistency::RadonIntermediate::PostProcess::Identity)
        .value("SquareRoot", EpipolarConsistency::RadonIntermediate::PostProcess::SquareRoot)
        .value("Logarithm", EpipolarConsistency::RadonIntermediate::PostProcess::Logarithm)
        .export_values();

    // EpipolarConsistencyRadonIntermediateCPU
    py::class_<EpipolarConsistency::MetricCPU>(m, "MetricCPU")
        .def(py::init<std::vector<Geometry::ProjectionMatrix>&, std::vector<EpipolarConsistency::RadonIntermediate*>, double>(), "Create MetricCPU instance.",
            py::arg("proj_mats"), py::arg("radon_derivatives"), py::arg("plane_angle_increment")=0.001745329251)
        .def("operator", py::overload_cast<>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for all image pairs on CPU")
        .def("operator", py::overload_cast<int>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for the i-th projection on CPU",
            py::arg("ref_projection"));

    py::class_<EpipolarConsistency::Metric>(m, "Metric");

    // EpipolarConsistencyRadonIntermediate (GPU)
    py::class_<EpipolarConsistency::MetricRadonIntermediate, EpipolarConsistency::Metric>(m, "MetricGPU")
        .def(py::init<>(), "Create empty MetricGPU object.")
        .def(py::init<const std::vector<Geometry::ProjectionMatrix>&, const std::vector<EpipolarConsistency::RadonIntermediate*>&>(), "Create MetricGPU objetc and set projection matrcies and radon intermediates.",
            py::arg("Ps"), py::arg("dtrs"))
        .def("evaluate", py::overload_cast<float*>(&EpipolarConsistency::MetricRadonIntermediate::evaluate), "Evaluates metric without any transformation of the geometry. Argument out has no function when set from python, don't use it.",
            py::arg("out")=0)
        .def("evaluate", py::overload_cast<const std::set<int>&, float*>(&EpipolarConsistency::MetricRadonIntermediate::evaluate), "Evaluates metric for just specific view. Argument out has no function when set from python, don't use it.",
            py::arg("views"), py::arg("out")=0)
        .def("evaluate", py::overload_cast<const std::vector<Eigen::Vector4i>&, float*>(&EpipolarConsistency::MetricRadonIntermediate::evaluate), "Evaluates metric without any transformation of the geometry. Indices addresses (P0,P1,dtr0,dtr1). Argument out has no function when set from python, don't use it.",
            py::arg("indices"), py::arg("out")=0)
        .def("setdKappa", &EpipolarConsistency::MetricRadonIntermediate::setdKappa, "Set plane angle increment dkappa.", 
            py::arg("dkappa")=0.001745329251)
        .def("setProjectionMatrices", &EpipolarConsistency::MetricRadonIntermediate::setProjectionMatrices, "Compute null space and pseudoinverse of projection matrices and convert to float (GPU is only faster for very large problems)",
            py::arg("Ps"));

}
