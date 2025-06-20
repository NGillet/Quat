import numpy as np

from quaternion import Quaternion


def test_quaternion_init():
    q = Quaternion(1, 2, 3, 4)
    assert np.allclose(q.w, [1])
    assert np.allclose(q.x, [2])
    assert np.allclose(q.y, [3])
    assert np.allclose(q.z, [4])


def test_quaternion_properties():
    q = Quaternion(1, 2, 3, 4)
    assert q.shape == 1
    assert q.copy != q
    assert isinstance(q.inverse, Quaternion)
    assert isinstance(q.conjugué, Quaternion)


def test_quaternion_norm():
    q = Quaternion(1, 2, 2, 1)
    assert np.allclose(q.norm, [3.16227766])


def test_quaternion_add_sub():
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(2, 1, 1, 0)
    q_sum = q1 + q2
    q_diff = q1 - q2
    assert np.allclose(q_sum.w, [3])
    assert np.allclose(q_diff.x, [1])
    assert np.allclose((2 + q1).w, [3])
    assert np.allclose((2 - q1).x, [0])


def test_quaternion_mul_scalar():
    q = Quaternion(1, 2, 3, 4)
    q2 = q * 2
    assert np.allclose(q2.w, [2])
    assert np.allclose(q2.x, [4])


def test_quaternion_mul_quaternion():
    q1 = Quaternion(1, 0, 1, 0)
    q2 = Quaternion(1, 0.5, 0.5, 0.75)
    q3 = q1 * q2
    assert isinstance(q3, Quaternion)
    assert q3.shape == 1


def test_quaternion_imul_scalar():
    q = Quaternion(1, 2, 3, 4)
    q *= 0.5
    assert np.allclose(q.w, [0.5])
    assert np.allclose(q.z, [2.0])


def test_quaternion_imul_quaternion():
    q1 = Quaternion(1, 0, 1, 0)
    q2 = Quaternion(1, 0, 1, 0)
    q1 *= q2
    assert isinstance(q1, Quaternion)


def test_quaternion_inverse():
    q = Quaternion(0, 1, 0, 0)
    identity = q * q.inverse
    assert np.allclose(identity.norm, [1])


def test_quaternion_getitem():
    q = Quaternion([1, 2], [0, 1], [0, 1], [1, 0])
    q0 = q[0]
    assert isinstance(q0, Quaternion)
    assert np.allclose(q0.w, [1])
    assert np.allclose(q0.z, [1])


def test_quaternion_array():
    q = Quaternion(1, 2, 3, 4)
    arr = np.array(q)
    assert arr.shape == (1, 4)


def test_quaternion_repr():
    q = Quaternion(1, 2, 3, 4)
    assert "Quaternion" in repr(q)
