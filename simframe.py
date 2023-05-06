from PIL import Image
import numpy as np
from skimage.measure import block_reduce
import cv2
import math

DEFAULT_CURRENT = 1  # Amps

# Magnetic permeability of free space
MU_0 = 1.25663706212e-6

# Relative resolution of the sampled b-field
# Higher number -> lower resolution
B_FIELD_SCALE_FAC = 1

# Size of the blocks to treat as single volume elements
# Higher number -> lower resolution
BLOCK_SIZE = 20

class SimFrame:
    def __init__(self, path, current=DEFAULT_CURRENT):
        self.path = path
        pic = Image.open(self.path, "r")
        self.arr = np.asarray(pic)[:,:,0] / 255
        self.arr = block_reduce(
            self.arr,
            block_size=(BLOCK_SIZE, BLOCK_SIZE),
            func=np.mean,
        )
        self.b_field = None
        self.current_density = None
        self.nocurrent = self.arr.sum() == 0
        if self.nocurrent:
            return
        cross_sectional_area = self.arr.sum() / self.arr.size
        self.current_density = current / cross_sectional_area

    def __str__(self):
        cd = f"J = {self.current_density}"
        return f"< SimFrame {self.arr.shape} with {cd} >"

    def bake_b_field(self):
        SCALE_FAC = B_FIELD_SCALE_FAC
        field = np.zeros((*( i // SCALE_FAC for i in self.arr.shape ), 2))
        for cx, row in enumerate(self.arr):
            for cy, prop_conductor in enumerate(row):
                pfx = f"({cx}, {cy}) -> "
                if prop_conductor == 0:
                    print(pfx + "Not conducting.")
                    continue
                if prop_conductor != 0:
                    print(pfx + "Calculating conductor contributions...")
                    cx_ctr = cx + (BLOCK_SIZE * SCALE_FAC) / 2
                    cy_ctr = cy + (BLOCK_SIZE * SCALE_FAC) / 2
                    for px in range(field.shape[0]):
                        for py in range(field.shape[1]):
                            jdv = np.array([
                                0.,
                                0.,
                                self.current_density,
                            ])
                            r = np.array([
                                cx_ctr - (px * SCALE_FAC),
                                cy_ctr - (py * SCALE_FAC),
                                0.,
                            ])
                            cross_product = np.cross(jdv, r)
                            dist = np.linalg.norm(r)
                            contribution = cross_product
                            contribution /= dist ** 3
                            contribution *= MU_0
                            contribution /= 4 * math.pi
                            contribution *= prop_conductor
                            if math.isnan(contribution[0]):
                                continue
                            if math.isnan(contribution[1]):
                                continue
                            field[px, py] += contribution[0:2]
        self.b_field = field

    def draw_b_field(self):
        image = cv2.imread(self.path)
        for ix, row in enumerate(self.b_field):
            for iy, vec in enumerate(row):
                x = ix * B_FIELD_SCALE_FAC * BLOCK_SIZE
                y = iy * B_FIELD_SCALE_FAC * BLOCK_SIZE
                x, y = int(x), int(y)
                vec_rescaled = (10, 10)
                startp = (x, y)
                endp = (x + int(vec_rescaled[0]), y + int(vec_rescaled[1]))
                color = (0, 0, 255)
                width = 2
                print(f"Drawing arrow at ({x}, {y})...")
                cv2.arrowedLine(image, startp, endp, color, width)
                print("ok")
        cv2.imshow("Testing: B-Field", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

test = SimFrame("frames/BadApple_358.jpg")

if __name__ == "__main__":
    test.bake_b_field()
    print("=== DONE CALCULATING B-FIELD!!! ===")
    test.draw_b_field()

