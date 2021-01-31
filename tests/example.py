import ecc
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
import matplotlib.pyplot as plt


# overview over all wrapped functions/ classes
print(help(ecc))

# create example projection
image = rescale(shepp_logan_phantom(), scale=0.125, multichannel=False).astype(np.float32)
# convert from numpy float32 to ecc internal data structure
ecc_image = ecc.ImageFloat2D(image)
# convert from internal data structure back to numpy
image_convert_back = np.array(ecc_image)

plt.figure()
ax1 = plt.subplot(121)
plt.imshow(image_convert_back, cmap='gray')
ax1.set_title('Input projection')

# compute radon intermediate with 100 detector elements and 200 angular steps
radon_intermediate = ecc.RadonIntermediate(ecc_image, 200, 100, ecc.RadonIntermediate.Filter.Derivative, ecc.RadonIntermediate.PostProcess.Identity)
# read back the radon intermediate from GPU to CPU
radon_intermediate.readback()
# convert radon intermediate back to numpy
radon_intermediate_convert_back = np.array(radon_intermediate)

ax2 = plt.subplot(122)
plt.imshow(radon_intermediate_convert_back, cmap='gray')
ax2.set_title('Radon intermediate')
plt.show()

# create two example projection matrices
R = np.eye(3)  # rotation
K = np.eye(3)  # intrinsics
t1 = [0, 0, 0]  # translation 1
t2 = [100, 0, 0]  # translation 2
P1 = ecc.makeProjectionMatrix(K, R, t1)
P2 = ecc.makeProjectionMatrix(K, R, t2)

# create ECC metric for a pair of images on GPU
metric_gpu = ecc.MetricGPU([P1, P2], [radon_intermediate, radon_intermediate])
# set the angle between epipolar planes (otherwise default value is taken)
metric_gpu = metric_gpu.setdKappa(0.001)
# evaluate the metric
out_gpu = metric_gpu.evaluate()
print(f'Result: {out_gpu}')