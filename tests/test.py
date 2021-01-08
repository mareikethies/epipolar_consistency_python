import ecc
import numpy as np

image1 = np.random.random((50, 50)).astype(np.float32)
image2 = np.random.random((50, 50)).astype(np.float32)

ecc_image1 = ecc.ImageFloat2D(image1)
ecc_image2 = ecc.ImageFloat2D(image2)

radon_intermediate1 = ecc.RadonIntermediate(ecc_image1, 1, 1, ecc.RadonIntermediate.Filter.Derivative, ecc.RadonIntermediate.PostProcess.Identity)
radon_intermediate2 = ecc.RadonIntermediate(ecc_image2, 1, 1, ecc.RadonIntermediate.Filter.Derivative, ecc.RadonIntermediate.PostProcess.Identity)



