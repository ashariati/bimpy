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

    A = list(reversed([[x, 2*lambda2] for x in np.linspace(-lambda1, lambda1, resolution + 2)[1:-1]]))
    B = [[lambda1, y] for y in np.linspace(0, 2*lambda2, resolution + 2)[1:-1]]

    v1 = np.array([lambda1, 0, z_ref])
    v2 = np.array([-lambda1, 0, z_ref])
    p1 = np.array([0, lambda2, z_ref])
    p2 = np.array([0, -lambda2, z_ref])

    # quadrant 1
    vertices_q1 = [v1]
    for i in range(resolution):

        px_top11 = np.linalg.det(np.array([v1[:2], A[i]]))
        px_top12 = np.linalg.det(np.array([[v1[0], 1], [A[i][0], 1]]))
        px_top21 = np.linalg.det(np.array([v2[:2], B[i]]))
        px_top22 = np.linalg.det(np.array([[v2[0], 1], [B[i][0], 1]]))

        py_top11 = np.linalg.det(np.array([v1[:2], A[i]]))
        py_top12 = np.linalg.det(np.array([[v1[1], 1], [A[i][1], 1]]))
        py_top21 = np.linalg.det(np.array([v2[:2], B[i]]))
        py_top22 = np.linalg.det(np.array([[v2[1], 1], [B[i][1], 1]]))

        bot11 = np.linalg.det(np.array([[v1[0], 1], [A[i][0], 1]]))
        bot12 = np.linalg.det(np.array([[v1[1], 1], [A[i][1], 1]]))
        bot21 = np.linalg.det(np.array([[v2[0], 1], [B[i][0], 1]]))
        bot22 = np.linalg.det(np.array([[v2[1], 1], [B[i][1], 1]]))
        denominator = np.linalg.det(np.array([[bot11, bot12], [bot21, bot22]]))

        px = np.linalg.det(np.array([[px_top11, px_top12], [px_top21, px_top22]])) / denominator
        py = np.linalg.det(np.array([[py_top11, py_top12], [py_top21, py_top22]])) / denominator

        vertices_q1.append(np.array([px, py, z_ref]))

    # quadrant 2
    vertices_q2 = [p1]
    for v1 in reversed(vertices_q1[1:]):
        vertices_q2.append(np.array([-v1[0], v1[1], z_ref]))

    # quadrant 3
    vertices_q3 = [v2]
    for v1 in vertices_q1[1:]:
        vertices_q3.append(np.array([-v1[0], -v1[1], z_ref]))

    # quadrant 4
    vertices_q4 = [p2]
    for v1 in reversed(vertices_q1[1:]):
        vertices_q4.append(np.array([v1[0], -v1[1], z_ref]))

    # vertices and edges
    vertices = vertices_q1 + vertices_q2 + vertices_q3 + vertices_q4
    edges = list(zip(range(len(vertices)), range(1, len(vertices))))
    edges.append((len(vertices) - 1, 0))
    edges = set(edges)

    return models.ConvexPolygon2D(vertices, edges)


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
