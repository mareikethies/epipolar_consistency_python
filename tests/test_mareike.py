import ecc
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale, radon
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
import json
from xml.dom import minidom


def test_radon_intermediate():
    # create shepp logan
    # image = rescale(shepp_logan_phantom(), scale=0.125, mode='reflect', multichannel=False).astype(np.float32)
    # image = np.concatenate((image, np.zeros((50, 10))), axis=1).astype(np.float32)
    image = shepp_logan_phantom()
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    sinogram_derived = np.diff(sinogram, axis=0)
    plt.figure()
    plt.imshow(sinogram, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.xlabel('Angles')
    # plt.ylabel('Virtual detector elements')
    plt.figure()
    plt.imshow(sinogram_derived, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.xlabel('Angles')
    # plt.ylabel('Virtual detector elements')
    plt.show()
    # convert from numpy to internal data structure
    ecc_image = ecc.ImageFloat2D(image)
    # convert from internal data structure back to numpy
    test_before = np.array(ecc_image)

    # compute radon intermediate
    radon_intermediate = ecc.RadonIntermediate(ecc_image, 100, 200, ecc.RadonIntermediate.Filter.Derivative, ecc.RadonIntermediate.PostProcess.Identity)
    # read back the derived forward projection from GPU to CPU
    radon_intermediate.readback()
    # convert radon intermediate back to numpy
    test_after = np.array(radon_intermediate)

    # plot before and after forward projection and derivation
    plt.figure()
    plt.subplot(131)
    plt.imshow(test_before, cmap="gray")
    plt.title('Input projection')
    plt.subplot(132)
    plt.imshow(sinogram, cmap="gray")
    plt.title('Derivative of sinogram')
    plt.subplot(133)
    plt.imshow(test_after, cmap="gray")
    plt.title('Radon Intermediate')
    plt.show()


def test_metric_cpu_vs_gpu(index1, index2):
    dkappa=0.001
    geometry_file = Path(r'simulated_projections_downsampled.txt')
    with open(geometry_file) as f:
        geometry_data = json.load(f)

    P1 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index1]),
                                  np.array(geometry_data['Rotations'][index1]),
                                  np.array(geometry_data['Translations'][index1]))
    P2 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index2]),
                                  np.array(geometry_data['Rotations'][index2]),
                                  np.array(geometry_data['Translations'][index2]))

    projections_file = Path(r'../../../Data/ERC/Ex-Vivo/Aging Kinetics/6522(7)_20201103/simulated_projections.tif')
    projections = imread(projections_file, plugin='tifffile')

    image1 = ecc.ImageFloat2D(projections[index1, :, :].astype(np.float32))
    image2 = ecc.ImageFloat2D(projections[index2, :, :].astype(np.float32))
    radon_intermediate1 = ecc.RadonIntermediate(image1, 100, 100,
                                                ecc.RadonIntermediate.Filter.Derivative,
                                                ecc.RadonIntermediate.PostProcess.Identity)
    radon_intermediate2 = ecc.RadonIntermediate(image2, 100, 100,
                                                ecc.RadonIntermediate.Filter.Derivative,
                                                ecc.RadonIntermediate.PostProcess.Identity)
    radon_intermediate1.readback()
    radon_intermediate2.readback()

    # on CPU
    metric_cpu = ecc.MetricCPU([P1, P2], [radon_intermediate1, radon_intermediate2], plane_angle_increment=dkappa)

    out_cpu = metric_cpu.operator()

    # on GPU
    metric_gpu = ecc.MetricGPU([P1, P2], [radon_intermediate1, radon_intermediate2])
    metric_gpu = metric_gpu.setdKappa(dkappa)

    out_gpu = metric_gpu.evaluate()

    # Andre says, it is ok if these are not the same as long as they have the same trend when plotted over one parameter
    print(f'CPU: {out_cpu}')
    print(f'GPU: {out_gpu}')

    return out_cpu, out_gpu


def plot_over_pairs():
    out_cpu = []
    out_gpu = []
    for i in range(1000):
        cpu, gpu = test_metric_cpu_vs_gpu(0, i)
        out_cpu.append(cpu)
        out_gpu.append(gpu)

    plt.figure()
    plt.plot(out_cpu)
    plt.figure()
    plt.plot(out_gpu)
    plt.show()


def test_real_data(index1, index2):
    geometry_file = Path(r'simulated_projections_downsampled.txt')
    with open(geometry_file) as f:
        geometry_data = json.load(f)

    P1 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index1]),
                                  np.array(geometry_data['Rotations'][index1]),
                                  np.array(geometry_data['Translations'][index1]))
    P2 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index2]),
                                  np.array(geometry_data['Rotations'][index2]),
                                  np.array(geometry_data['Translations'][index2]))

    projections_file = Path(r'../../../Data/ERC/Ex-Vivo/Aging Kinetics/6522(7)_20201103/simulated_projections.tif')
    projections = imread(projections_file, plugin='tifffile')

    image1 = ecc.ImageFloat2D(projections[index1, :, :].astype(np.float32))
    image2 = ecc.ImageFloat2D(projections[index2, :, :].astype(np.float32))
    radon_intermediate1 = ecc.RadonIntermediate(image1, 100, 100,
                                                ecc.RadonIntermediate.Filter.Derivative,
                                                ecc.RadonIntermediate.PostProcess.Identity)
    radon_intermediate2 = ecc.RadonIntermediate(image2, 100, 100,
                                                ecc.RadonIntermediate.Filter.Derivative,
                                                ecc.RadonIntermediate.PostProcess.Identity)
    radon_intermediate1.readback()
    radon_intermediate2.readback()

    plt.figure()
    plt.subplot(221)
    plt.imshow(np.array(image1))
    plt.subplot(222)
    plt.imshow(np.array(image2))
    plt.subplot(223)
    plt.imshow(np.array(radon_intermediate1))
    plt.subplot(224)
    plt.imshow(np.array(radon_intermediate2))
    plt.show()

    metric = ecc.MetricCPU([P1, P2], [radon_intermediate1, radon_intermediate2])

    out = metric.operator()

    print(out)


def get_projection_matrices_from_conrad_xml():
    conrad_xml = minidom.parse('/home/mareike/Data/SimplePhantom/ConradSheppLogan3DForward.xml')
    all = conrad_xml.getElementsByTagName("object")

    mats = []
    for element in all:
        if element.getAttribute('class') == 'edu.stanford.rsl.conrad.geometry.Projection':
            pm_str = element.childNodes[1].childNodes[1].childNodes[0].data
            pm_str = pm_str.replace('[', '')
            pm_str = pm_str.replace(']', '')
            pm_str = pm_str.replace(';', '')
            parts = pm_str.split(' ')
            mat = np.zeros((3, 4))
            mat[0, 0] = eval(parts[0])
            mat[0, 1] = eval(parts[1])
            mat[0, 2] = eval(parts[2])
            mat[0, 3] = eval(parts[3])
            mat[1, 0] = eval(parts[4])
            mat[1, 1] = eval(parts[5])
            mat[1, 2] = eval(parts[6])
            mat[1, 3] = eval(parts[7])
            mat[2, 0] = eval(parts[8])
            mat[2, 1] = eval(parts[9])
            mat[2, 2] = eval(parts[10])
            mat[2, 3] = eval(parts[11])

            mats.append(mat)
    return np.moveaxis(np.array(mats), 0, 2)


def compute_radon_intermediates(projection_images, size_alpha=100, size_t=100):
    radon_intermediates = []
    for image in projection_images:
        image = ecc.ImageFloat2D(image.astype(np.float32))
        radon_intermediate = ecc.RadonIntermediate(image, size_alpha, size_t,
                                                   ecc.RadonIntermediate.Filter.Derivative,
                                                   ecc.RadonIntermediate.PostProcess.Identity)
        radon_intermediate.readback()
        radon_intermediates.append(radon_intermediate)
    return radon_intermediates


def test_more_pairs(number):
    projection_matrices = get_projection_matrices_from_conrad_xml()
    projection_matrices = [projection_matrices[:, :, i] for i in range(projection_matrices.shape[2])]
    projection_images = imread('/home/mareike/Data/SimplePhantom/SheppLogan3DForward.tif', plugin='tifffile')
    projection_images = [projection_images[i, :, :] for i in range(projection_images.shape[0])]

    radon_intermediates = compute_radon_intermediates(projection_images[:number])
    metric_gpu = ecc.MetricGPU(projection_matrices[:number], radon_intermediates)
    out = metric_gpu.evaluate()
    print(out)


if __name__ == "__main__":
    test_radon_intermediate()



