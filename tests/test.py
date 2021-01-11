import ecc
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
import json


def test_radon_intermediate():
    # create shepp logan
    image = rescale(shepp_logan_phantom(), scale=0.125, mode='reflect', multichannel=False).astype(np.float32)
    # convert from numpy to internal data structure
    ecc_image = ecc.ImageFloat2D(image)
    # convert from internal data structure back to numpy
    test_before = np.array(ecc_image)

    # compute radon intermediate
    radon_intermediate = ecc.RadonIntermediate(ecc_image, 100, 100, ecc.RadonIntermediate.Filter.Derivative, ecc.RadonIntermediate.PostProcess.Identity)
    # read back the derived forward projection from GPU to CPU
    radon_intermediate.readback()
    # convert radon intermediate back to numpy
    test_after = np.array(radon_intermediate)

    # plot before and after forward projection and derivation
    plt.figure()
    plt.subplot(121)
    plt.imshow(test_before, cmap="gray")
    plt.title('Input projection')
    plt.subplot(122)
    plt.imshow(test_after, cmap="gray")
    plt.title('Derivative of sinogram')
    plt.show()


def test_example_data(index1, index2):
    geometry_file = Path(r'simulated_projections_downsampled.txt')
    with open(geometry_file) as f:
        geometry_data = json.load(f)

    P1 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index1]),
                                  np.array(geometry_data['Rotations'][index1]),
                                  np.array(geometry_data['Translations'][index1]))
    P2 = ecc.makeProjectionMatrix(np.array(geometry_data['Intrinsics'][index2]),
                                  np.array(geometry_data['Rotations'][index2]),
                                  np.array(geometry_data['Translations'][index2]))

    projections_file = Path(r'../../../Data/ERC/Aging Kinetics/6522(7)_20201103/simulated_projections.tif')
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


if __name__ == "__main__":
    test_example_data(0, 100)



