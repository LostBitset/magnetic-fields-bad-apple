from PIL import Image
import numpy as np
from skimage.measure import block_reduce
import cv2
import math

DEFAULT_CURRENT = 1  # Amps

# Magnetic permeability of free space
MU_0 = 1.25663706212e-6

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
        cross_sectional_area = self.arr.sum() / self.arr.size
        print(f"[info] Cross sectional area: {cross_sectional_area}")
        self.nocurrent = cross_sectional_area < 0.00001
        if self.nocurrent:
            print("[info] NO CURRENT")
            return
        self.current_density = current / cross_sectional_area

    def __str__(self):
        cd = f"J = {self.current_density}"
        return f"< SimFrame {self.arr.shape} with {cd} >"

    def bake_b_field(self):
        if self.nocurrent:
            return
        field = np.zeros((*self.arr.shape, 2))
        for cx, row in enumerate(self.arr):
            for cy, prop_conductor in enumerate(row):
                pfx = f"({cx}, {cy}) -> "
                if prop_conductor == 0:
                    print(pfx + "Not conducting.")
                    continue
                if prop_conductor != 0:
                    print(pfx + "Calculating conductor contributions...")
                    for px in range(field.shape[0]):
                        for py in range(field.shape[1]):
                            jdv = np.array([
                                0.,
                                0.,
                                self.current_density,
                            ])
                            r = np.array([
                                cx - px,
                                cy - py,
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
        if self.nocurrent:
            return image
        norms = np.zeros(self.b_field.shape[:2])
        for ix, row in enumerate(self.b_field):
            for iy, vec_raw in enumerate(row):
                norms[ix][iy] = np.linalg.norm(vec_raw * 1e8)
        for ix, row in enumerate(self.b_field):
            for iy, vec_raw in enumerate(row):
                x = ix * BLOCK_SIZE
                y = iy * BLOCK_SIZE
                x, y = int(x), int(y)
                vec = vec_raw * 1e8
                vec_norm = norms[ix][iy]
                vec_unit = vec / vec_norm
                startp = (x, y)
                endp = (
                    x + int(vec_unit[0] * BLOCK_SIZE),
                    y + int(vec_unit[1] * BLOCK_SIZE),
                )
                color = (0, 0, 255)
                width = 2
                alpha = vec_norm / norms.max()
                print(f"Drawing arrow at ({x}, {y})...")
                over = image.copy()
                cv2.arrowedLine(over, startp[::-1], endp[::-1], color, width)
                image = cv2.addWeighted(over, alpha, image, 1 - alpha, 0)
                print("ok")
        print("Created image with B-field. Returning it.")
        return image

test = SimFrame("frames/BadApple_810.jpg")

if __name__ == "__main__":
    test.bake_b_field()
    print("=== DONE CALCULATING B-FIELD!!! ===")
    image = test.draw_b_field()
    cv2.imshow("Test image with B-field", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

