import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = Image.open('../data/simple_teapot/teapot_.VRayNormals.jpg')
np_img = np.array(img)/255.0
# print(np_img)

normal_vectors = np_img * 2 - 1
print(np_img[0,0])

plot = plt.imshow(normal_vectors)
plt.show()
