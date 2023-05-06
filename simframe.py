from PIL import Image
import numpy as np

class SimFrame:
    def __init__(self, path):
        pic = Image.open(path, "r")
        self.arr = np.asarray(pic)[:,:,0] > 200

print(SimFrame("frames/BadApple_358.jpg").arr)

