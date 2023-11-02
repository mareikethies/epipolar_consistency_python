import ecc
import numpy as np


# overview over all wrapped functions/ classes
print(help(ecc))

# set parameters for computation of Radon intermediates
size_alpha = 200
size_t = 100

# load example projection images
projection_images = np.load('../example_data/sinogram_cone.npy').astype(np.float32)
# convert to list
projection_images = [projection_images[i, :, :] for i in range(projection_images.shape[0])]
# compute radon intermediates with 100 detector elements and 200 angular steps
radon_intermediates = []
for image in projection_images:
    image = ecc.ImageFloat2D(image.astype(np.float32))
    radon_intermediate = ecc.RadonIntermediate(image, size_alpha, size_t,
                                               ecc.RadonIntermediate.Filter.Derivative,
                                               ecc.RadonIntermediate.PostProcess.Identity)
    radon_intermediates.append(radon_intermediate)

# load example projection matrices
projection_matrices = np.load('../example_data/projection_matrices.npy')
# convert to list
projection_matrices = [projection_matrices[i, :, :] for i in range(projection_matrices.shape[0])]

# create object to evaluate ECC metric on GPU
ecc_metric_gpu = ecc.MetricGPU(projection_matrices, radon_intermediates)

# set the angle between epipolar planes (otherwise default value is taken)
metric_gpu = ecc_metric_gpu.setdKappa(0.001)
# evaluate the metric
out_gpu = metric_gpu.evaluate()
print(f'Result: {out_gpu}')

