import numpy as np

def discrete_phantom(xcoord, ycoord, phantomE, phantomC):
    image = np.zeros(xcoord.shape)
    nclipinfo = 0

    # Loop over each row in phantomE
    for k in range(phantomE.shape[0]):
        Vx0 = np.array([xcoord.ravel() - phantomE[k, 0], ycoord.ravel() - phantomE[k, 1]])
        D = np.array([[1 / phantomE[k, 2], 0], [0, 1 / phantomE[k, 3]]])
        phi = phantomE[k, 4] * np.pi / 180
        Q = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        f = phantomE[k, 5]
        nclip = int(phantomE[k, 6])

        # Calculate the ellipse equation and find indices inside the ellipse
        equation1 = np.sum((D @ Q @ Vx0)**2, axis=0)
        i = np.where(equation1 <= 1.0)[0]

        # Handle clipping if nclip > 0
        if nclip > 0:
            for j in range(nclip):
                nclipinfo += 1
                d = phantomC[0, nclipinfo - 1]
                equation2 = np.dot([phantomC[2, nclipinfo - 1], phantomC[3, nclipinfo - 1]], Vx0)
                i = i[np.where(equation2[i] < d)]

        image.ravel()[i] += f

    return image


def bilinear_value_pixel(x, y, pmap):
    x1 = np.int(np.floor(x))
    x2 = np.int(np.ceil(x))
    y1 = np.int(np.floor(y))
    y2 = np.int(np.ceil(y))

    return pmap[x1][y1] * (x2-x) * (y2-y) + pmap[x1][y2] * (x2-x) * (y-y1) + pmap[x2][y1] * (x-x1) * (y2-y) + pmap[x2][y2] * (x-x1) * (y-y1)

def evaluate(xcoords, ycoords, dx, dy, func):
    res = 0.0

    

    return res

if __name__ == "__main__":
    x, step = np.linspace(-50, 50, endpoint=False, num=1000, retstep=True, dtype=np.float64)
    x += step

    xcoords, ycoords = np.meshgrid(x, np.flip(x))
    base = discrete_phantom(xcoords, ycoords)
    base = base.reshape((1000, 1000))

    x, step = np.linspace(0, 100, endpoint=False, num=1000, retstep=True, dtype=np.float64)
    x += step

    xcoords, ycoords = np.meshgrid(x, np.flip(x))