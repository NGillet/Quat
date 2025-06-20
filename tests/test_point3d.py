import numpy as np
from numpy.testing import assert_allclose

from quaternion import Point3D


def test_basic_construction():
    p = Point3D(1.0, 2.0, 3.0)
    assert_allclose(p.x, [1.0])
    assert_allclose(p.y, [2.0])
    assert_allclose(p.z, [3.0])


def test_vectorial_construction():
    p = Point3D(np.array([1.0, 4.0]), np.array([2.0, 5.0]), np.array([3.0, 6.0]))
    assert_allclose(p.x, [1.0, 4.0])
    assert_allclose(p.y, [2.0, 5.0])
    assert_allclose(p.z, [3.0, 6.0])


def test_addition():
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = Point3D(4.0, 5.0, 6.0)
    p3 = p1 + p2
    assert_allclose(p3.x, [5.0])
    assert_allclose(p3.y, [7.0])
    assert_allclose(p3.z, [9.0])


def test_inplace_addition():
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = Point3D(4.0, 5.0, 6.0)
    p1 += p2
    assert_allclose(p1.x, [5.0])
    assert_allclose(p1.y, [7.0])
    assert_allclose(p1.z, [9.0])


def test_subtraction():
    p1 = Point3D(4.0, 5.0, 6.0)
    p2 = Point3D(1.0, 2.0, 3.0)
    p3 = p1 - p2
    assert_allclose(p3.x, [3.0])
    assert_allclose(p3.y, [3.0])
    assert_allclose(p3.z, [3.0])


def test_inplace_subtraction():
    p1 = Point3D(4.0, 5.0, 6.0)
    p2 = Point3D(1.0, 2.0, 3.0)
    p1 -= p2
    assert_allclose(p1.x, [3.0])
    assert_allclose(p1.y, [3.0])
    assert_allclose(p1.z, [3.0])


def test_or_concatenation():
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = Point3D(4.0, 5.0, 6.0)
    p3 = p1 | p2
    assert_allclose(p3.x, [1.0, 4.0])
    assert_allclose(p3.y, [2.0, 5.0])
    assert_allclose(p3.z, [3.0, 6.0])


def test_getitem():
    p = Point3D(np.array([1.0, 4.0]), np.array([2.0, 5.0]), np.array([3.0, 6.0]))
    p0 = p[0]
    assert_allclose(p0.x, [1.0])
    assert_allclose(p0.y, [2.0])
    assert_allclose(p0.z, [3.0])


def test_copy():
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = p1.copy
    assert_allclose(p2.x, p1.x)
    assert_allclose(p2.y, p1.y)
    assert_allclose(p2.z, p1.z)
