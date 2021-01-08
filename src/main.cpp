#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> //for automatic type conversion of Eigen to/from python
#include <pybind11/stl.h> // for automatic type conversion of STL containers to/from python

#include "LibProjectiveGeometry/ProjectionMatrix.h"
#include "LibEpipolarConsistency/EpipolarConsistencyRadonIntermediateCPU.hxx"
#include "NRRD/nrrd_image.hxx"
#include "NRRD/nrrd_image_view.hxx"

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
    //TODO: why is python unhappy with normalizeProjectionMatrix (fails at import)??
    //m.def("normalizeProjectionMatrix", &Geometry::normalizeProjectionMatrix, 
    //    "Normalize a camera matrix P=[M|p4] by -sign(det(M))/||m3|| such that (du,dv,d)'=P(X,Y,Z,1)' encodes the depth d.",
    //    py::arg("P"));
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

    py::class_<NRRD::ImageView<float>>(m, "ImageViewFloat");

    //templates need to be set to a specific data type for wrapping
    //base classes need to be wrapped, too. Otherwise python will complain about an unknown base class when importing the module.
    //pointers to data larger than one value in C++ need to be wraped using py::buffer
    py::class_<NRRD::Image<float>, NRRD::ImageView<float>>(m, "ImageFloat", py::buffer_protocol())
        .def(py::init<int, int, int, float*>(), "Create a 3D image without data pointer.",
            py::arg("w"), py::arg("h"), py::arg("d")=1, py::arg("dt")=0x0)
        .def(py::init([](py::buffer const b){

            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<float>::format())
                throw std::runtime_error("Buffer must be float32.");
            if (info.ndim != 2)
                throw std::runtime_error("Buffer must have 2 dimensions.");

            auto v = new NRRD::Image<float>(info.shape[0], info.shape[1], 1);
            memcpy(v->data, info.ptr, sizeof(float) * (size_t) (v->size(0) * v->size(1)));
            return v;
        }), "Create a 2D image from a python buffer.", py::arg("b"))
        /// Provide buffer access
       .def_buffer([](NRRD::Image<float> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data,                               /* Pointer to buffer */
                { m.size(0), m.size(1) },                 /* Buffer dimensions */
                { sizeof(float) * size_t(m.size(1)),     /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    // EpipolarConsistencyRadonIntermediateCPU
    py::class_<EpipolarConsistency::MetricCPU>(m, "MetricCPU")
        .def(py::init<std::vector<Geometry::ProjectionMatrix>&, std::vector<EpipolarConsistency::RadonIntermediate*>, double>(), "Create MetricCPU instance.",
            py::arg("proj_mats"), py::arg("radon_derivatives"), py::arg("plane_angle_increment")=0.001745329251)
        .def("operator", py::overload_cast<>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for all image pairs on CPU")
        .def("operator", py::overload_cast<int>(&EpipolarConsistency::MetricCPU::operator()), "Evaluates consistency for the i-th projection on CPU",
            py::arg("ref_projection"));

}
