import numpy as np
import models


def rectangle(plane, x, y, w, h):

    assert isinstance(plane, models.Plane), "Expected plane to by of type Plane, got %s instead" % type(plane)

    distance = -plane.coefficients[3]

    vertices = np.array([[distance, w / 2, h / 2],
                         [distance, -(w / 2), h / 2],
                         [distance, -(w / 2), -(h / 2)],
                         [distance, w / 2, -(h / 2)]])
    vertices = vertices + np.array([0, x, y])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    v1 = plane.coefficients[:3]
    if not np.isclose(v1[2], 1):
        v2 = np.array([-v1[1], v1[0], 0])
    else:
        v2 = np.array([0, v1[2], -v1[1]])
    v3 = np.cross(v1, v2)

    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    R = np.array([v1, v2, v3]).T
    vertices = np.dot(R, vertices.T).T

    return models.Polygon3D(vertices, edges, plane)


def ellipse(lambda1, lambda2, resolution, z_ref=0):

    vertices = [[lambda1, 0], [-lambda1, 0], [0, lambda2], [0, -lambda2]]
    for x in np.linspace(0, lambda1, resolution + 2)[1:-1]:
        y = (lambda2 / lambda1) * np.sqrt(lambda1 ** 2 - x ** 2)
        vertices.append([x, y])
        vertices.append([-x, y])
        vertices.append([-x, -y])
        vertices.append([x, -y])
    vertices = np.array(vertices)

    vertices3 = np.zeros((vertices.shape[0], 3))
    vertices3[:, :2] = vertices
    vertices3[:, 2] = z_ref

    return models.ConvexPolygon2D(vertices3, []).sortccw()


def random_evidence(n, z_ref, max_radius, min_range, max_range, resolution=4):

    min_range = np.array(min_range) + max_radius
    max_range = np.array(max_range) - max_radius

    evidence = []
    for i in range(n):
        lambda1 = max_radius * min(np.random.rand() + 0.1, 1)
        lambda2 = max_radius * min(np.random.rand() + 0.1, 1)

        lambda1, lambda2 = (lambda2, lambda1) if lambda2 > lambda1 else (lambda1, lambda2)

        e = ellipse(lambda1, lambda2, resolution, z_ref)

        v1 = (np.random.rand(2) - 0.5)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.array([-v1[1], v1[0]])

        R = np.array([v1, v2]).T
        t = ((max_range - min_range) * np.random.rand(2)) + min_range
        e.rigid(R, t)

        evidence.append(e)

    return evidence


def random_planes(num_planes, max_distance, a=None, b=None, c=None):

    coefs = np.random.rand(num_planes, 4) - 0.5

    if a is not None:
        coefs[:, 0] = a
    if b is not None:
        coefs[:, 1] = b
    if c is not None:
        coefs[:, 2] = c

    coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
    coefs[:, 3] = 2 * max_distance * coefs[:, 3]

    return [models.Plane(c) for c in coefs]
