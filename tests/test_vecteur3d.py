from numpy.testing import assert_allclose

from quaternion import Point3D, Vecteur3D


def test_basic_properties():
    pa = Point3D(1, 2, 3)
    vo = Vecteur3D(pa)

    assert vo.shape == 1
    assert_allclose(vo.v.point, [[1, 2, 3]])


def test_add_sub():
    a = Vecteur3D(Point3D(1, 0, 0))
    b = Vecteur3D(Point3D(0, 2, 0))

    c = a + b
    assert_allclose(c.v.point, [[1, 2, 0]])

    d = c - a
    assert_allclose(d.v.point, [[0, 2, 0]])


def test_inplace_operations():
    v = Vecteur3D(Point3D(1, 1, 1))
    v += Vecteur3D(Point3D(1, 0, 0))
    assert_allclose(v.v.point, [[2, 1, 1]])

    v -= Vecteur3D(Point3D(0, 1, 0))
    assert_allclose(v.v.point, [[2, 0, 1]])


def test_unitaire():
    v = Vecteur3D(Point3D(3, 0, 0))
    u = v.unitaire
    assert_allclose(u.v.point, [[1, 0, 0]])


def test_vector_indexing():
    V = Vecteur3D(Point3D([1, 2], [0, 0], [0, 0]))
    assert V[0].shape == 1
    assert_allclose(V[1].v.point, [[2, 0, 0]])
