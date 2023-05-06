from PIL import Image
import numpy as np
from skimage.measure import block_reduce
import math

DEFAULT_CURRENT = 1  # Amps

# Magnetic permeability of free space
MU_0 = 1.25663706212e-6

class SimFrame:
    def __init__(self, path, current=DEFAULT_CURRENT):
        pic = Image.open(path, "r")
        self.arr = np.asarray(pic)[:,:,0]
        self.arr = block_reduce(self.arr, block_size=(5, 5), func=np.mean)
        self.arr = self.arr > 200
        self.nocurrent = self.arr.sum() == 0
        if self.nocurrent:
            self.current_density = None
            return
        cross_sectional_area = self.arr.sum() / self.arr.size
        self.current_density = current / cross_sectional_area

    def __str__(self):
        cd = f"J = {self.current_density}"
        return f"< SimFrame {self.arr.shape} with {cd} >"

    def b_field(self):
        SCALE_FAC = 4
        field = np.zeros((*( i // SCALE_FAC for i in self.arr.shape ), 2))
        for cx, row in enumerate(self.arr):
            for cy, is_conductor in enumerate(row):
                pfx = f"({cx}, {cy}) -> "
                if not is_conductor:
                    print(pfx + "Not conducting.")
                if is_conductor:
                    print(pfx + "Calculating conductor contributions...")
                    for px in range(field.shape[0]):
                        for py in range(field.shape[1]):
                            jdv = np.array([
                                0.,
                                0.,
                                self.current_density,
                            ])
                            r = np.array([
                                cx - (px * SCALE_FAC),
                                cy - (py * SCALE_FAC),
                                0.,
                            ])
                            cross_product = np.cross(jdv, r)
                            dist = np.linalg.norm(r)
                            contribution = cross_product
                            contribution /= dist ** 3
                            contribution *= MU_0
                            contribution /= 4 * math.pi
                            if math.isnan(contribution[0]):
                                continue
                            if math.isnan(contribution[1]):
                                continue
                            field[px, py] += contribution[0:2]
        return field

test = SimFrame("frames/BadApple_358.jpg")

if __name__ == "__main__":
    print(test)
    print(test.arr)

