import ecc
import numpy as np

test_image = np.random.random((50, 50))
test_image = test_image.astype(np.float32)

ecc_image = ecc.ImageFloat(test_image)

print(ecc_image)

