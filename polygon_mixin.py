import numpy as np


class PolygonMixin:

    def round_off(self, number, ndigits=0):
        p = 10 ** ndigits
        return (number * p * 2 + 1) // 2 / p

    def unique_close(self, vertices, atol=1e-12):
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            if np.all(np.isclose(v1, v2, atol=atol)):
                continue

            yield v1

        yield vertices[-1]