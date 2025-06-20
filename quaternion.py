from typing import Union

import numpy as np

eps = 1.0e-9
##### --- #####


class Quaternion:
    def __init__(
        self,
        w: Union[float, np.ndarray],
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._w = self._ensure_column_vector(w)
        self._point = np.stack(
            [
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
            ],
            axis=-1,
        )
        if self._point.ndim == 1:
            self._point = self._point[None, :]

    # --- Utilities ---
    def _ensure_column_vector(self, arr: Union[float, np.ndarray]) -> np.ndarray:
        arr = np.atleast_1d(arr)
        return np.asarray(arr, dtype=np.float64).reshape(-1, 1)

    def _is_scalar_or_1d(self, q) -> bool:
        return np.isscalar(q) or (
            isinstance(q, (np.ndarray, list)) and np.array(q).ndim <= 1
        )

    def _binary_op(self, other, op):
        if isinstance(other, Quaternion):
            return Quaternion(
                op(self._w, other._w),
                op(self.x, other.x),
                op(self.y, other.y),
                op(self.z, other.z),
            )
        elif self._is_scalar_or_1d(other):
            return Quaternion(
                op(self._w, other),
                op(self.x, other),
                op(self.y, other),
                op(self.z, other),
            )
        return NotImplemented

    # --- Properties ---
    @property
    def w(self):
        return self._w[:, 0]

    @property
    def x(self):
        return self._point[:, 0]

    @property
    def y(self):
        return self._point[:, 1]

    @property
    def z(self):
        return self._point[:, 2]

    @property
    def point(self):
        return self._point

    @property
    def norm(self):
        return np.linalg.norm(np.concatenate([self._w, self._point], axis=1), axis=1)

    @property
    def conjugué(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @property
    def inverse(self):
        return self.conjugué * (1.0 / self.norm**2)

    @property
    def copy(self):
        return Quaternion(self.w, self.x, self.y, self.z)

    @property
    def shape(self):
        return self._w.shape[0]

    def __array__(self, dtype=None, copy=True):
        arr = np.concatenate([self._w, self._point], axis=1)
        if dtype is not None:
            arr = arr.astype(dtype, copy=copy)
        elif not copy:
            arr = np.array(arr, copy=False)  # safe fallback
        return arr

    # --- Arithmetic ---
    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return Quaternion(other, other, other, other) - self

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            A, B = self, q
            w1 = A._w[:, 0][:, np.newaxis]  ### (N,1)
            w2 = B._w[:, 0][np.newaxis, :]  ### (1,M)
            p1 = A._point[:, np.newaxis, :]  ### (N,1,3)
            p2 = B._point[np.newaxis, :, :]  ### (1,M,3)

            a = w1 * w2 - np.sum(p1 * p2, axis=2)  # (N,M)
            v = (
                w1[..., np.newaxis] * p2 + w2[..., np.newaxis] * p1 + np.cross(p1, p2)
            )  # (N,M,3)

            return Quaternion(
                a.reshape(-1),
                v[..., 0].reshape(-1),
                v[..., 1].reshape(-1),
                v[..., 2].reshape(-1),
            )

        elif self._is_scalar_or_1d(q):
            return Quaternion(self._w * q, self.x * q, self.y * q, self.z * q)

    def __rmul__(self, q):
        return self.__mul__(q)

    def __imul__(self, q):
        if isinstance(q, Quaternion):
            A, B = self, q
            w1 = A._w[:, 0][:, np.newaxis]  ### (N,1)
            w2 = B._w[:, 0][np.newaxis, :]  ### (1,M)
            p1 = A._point[:, np.newaxis, :]  ### (N,1,3)
            p2 = B._point[np.newaxis, :, :]  ### (1,M,3)

            a = w1 * w2 - np.sum(p1 * p2, axis=2)  ### (N,M)
            v = (
                w1[..., np.newaxis] * p2 + w2[..., np.newaxis] * p1 + np.cross(p1, p2)
            )  # (N,M,3)

            self._w = a.reshape(-1)  ### (NxM)
            self._point = v.reshape([-1, 3])  ### (NxM,3)
            return self

        elif self._is_scalar_or_1d(q):
            print("toto")
            self._w *= q
            self._point *= q

            return self
        return NotImplemented

    def __getitem__(self, key):
        return Quaternion(self.w[key], self.x[key], self.y[key], self.z[key])

    def __repr__(self):
        return f"Quaternion(w={self.w.tolist()}, x={self.x.tolist()}, y={self.y.tolist()}, z={self.z.tolist()})"


class Point3D(Quaternion):
    def __init__(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
        **kwargs,
    ):
        super().__init__(w=np.zeros_like(x), x=x, y=y, z=z, **kwargs)

    @property
    def copy(self) -> "Point3D":
        return self.__class__(*self.point.T)

    def __array__(self, dtype=None, copy=True):
        arr = self._point
        if dtype is not None:
            arr = arr.astype(dtype, copy=copy)
        elif not copy:
            arr = np.array(arr, copy=False)  # safe fallback
        return arr

    def __add__(self, q) -> "Point3D":
        return self.__class__(*super().__add__(q).point.T)

    def __iadd__(self, q) -> "Point3D":
        result = super().__add__(q)
        self._w = result._w
        self._point = result._point
        return self

    def __sub__(self, q) -> "Point3D":
        return self.__class__(*super().__sub__(q).point.T)

    def __isub__(self, q) -> "Point3D":
        result = super().__sub__(q)
        self._w = result._w
        self._point = result._point
        return self

    def __mul__(self, r) -> "Point3D":
        if hasattr(
            r, "rot_quat_form"
        ):  ### that is a way to select only Rotation without direct call
            rot = r.rot_quat_form
            rot_inv = r.rot_quat_form.inverse
            new_self = ((rot * (self - r.vect_dir.po)) * rot_inv) + r.vect_dir.po
            return self.__class__(*new_self._point.T)
        else:
            new_Q = super().__mul__(r)
            return self.__class__(*new_Q._point.T)

    def __imul__(self, r) -> "Point3D":
        if hasattr(
            r, "rot_quat_form"
        ):  ### that is a way to select only Rotation without direct call
            rot = r.rot_quat_form
            rot_inv = r.rot_quat_form.inverse
            new_self = ((rot * (self - r.vect_dir.po)) * rot_inv) + r.vect_dir.po
            self._w = new_self._w
            self._point = new_self._point
            return self
        else:
            super().__imul__(r)
            return self

    def __irshift__(self, v):
        if hasattr(
            v, "v"
        ):  ### that is a way to select only Vecteur3D without direct call
            self += v.v
            return self
        return NotImplemented

    def __or__(self, p: "Point3D") -> "Point3D":
        """
        Concatenate differents instances of Point3D into a new vectorial Point3D
        """
        if isinstance(p, Point3D):
            return self.__class__(*np.concatenate([self.point, p.point]).T)

        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x.tolist()}, y={self.y.tolist()}, z={self.z.tolist()})"

    def __getitem__(self, key):
        return self.__class__(self.x[key], self.y[key], self.z[key])


class Vecteur3D:
    def __init__(
        self,
        point_arrivee: "Point3D",
        point_origine: "Point3D" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pa = point_arrivee
        if point_origine is None:
            ### Defaul Origin (0,0,0)
            self.po = Point3D(*np.zeros_like(self.pa).T)
        else:
            self.po = point_origine

        assert self.po.shape == self.pa.shape

    def _is_scalar_or_1d(self, q) -> bool:
        return np.isscalar(q) or (
            isinstance(q, (np.ndarray, list)) and np.array(q).ndim <= 1
        )

    @property
    def v(self):
        ### Point3D representing the vector
        return self.pa - self.po

    @property
    def shape(self):
        return self.po.shape

    @property
    def unitaire(self) -> "Vecteur3D":
        pa_new = self.po + Point3D(*(self.v.point.T / self.v.norm))
        return self.__class__(pa_new, point_origine=self.po)

    @property
    def to_unitaire(
        self,
    ) -> "Vecteur3D":
        new_v = (self.v.point.T / self.v.norm).T
        self.pa = self.po + Point3D(*new_v.T)
        return self

    @property
    def copy(self) -> "Vecteur3D":
        return self.__class__(self.pa, self.po)

    @property
    def orthogonal(self):
        """
        Get an arbitrary orthogonal vector

        example:
        V = Vx, Vy, Vz
        an orthogonal vector is : -(Vy+Vz) / Vx , 1 , 1 ( for Vx !=0)
        """
        vx, vy, vz = self.v.x, self.v.y, self.v.z

        new_x = np.full_like(vx, np.nan)
        new_y = np.full_like(vy, np.nan)
        new_z = np.full_like(vz, np.nan)

        ### condition: x ≠ 0
        cond_x = np.abs(vx) > eps
        new_x[cond_x] = -(vy[cond_x] + vz[cond_x]) / vx[cond_x]
        new_y[cond_x] = 1.0
        new_z[cond_x] = 1.0

        ### y ≠ 0 and x == 0
        cond_y = np.logical_and(~cond_x, np.abs(vy) > eps)
        new_x[cond_y] = 1.0
        new_y[cond_y] = -(vx[cond_y] + vz[cond_y]) / vy[cond_y]
        new_z[cond_y] = 1.0

        # z ≠ 0 and x == 0 and y == 0
        cond_z = np.logical_and(~cond_x, ~cond_y)
        cond_z = np.logical_and(cond_z, np.abs(vz) > eps)
        new_x[cond_z] = 1.0
        new_y[cond_z] = 1.0
        new_z[cond_z] = -(vx[cond_z] + vy[cond_z]) / vz[cond_z]

        ### Fallback (all components zero): fill with [0, 0, 0] (arbitrary)
        fallback = ~(cond_x | cond_y | cond_z)
        new_x[fallback] = 0.0
        new_y[fallback] = 0.0
        new_z[fallback] = 0.0

        orthogonal_point = Point3D(new_x, new_y, new_z) + self.po
        return Vecteur3D(orthogonal_point, point_origine=self.po)

    def __or__(self, v: "Vecteur3D") -> "Vecteur3D":
        if isinstance(v, Vecteur3D):
            po_new = self.po | v.po
            pa_new = self.pa | v.pa
            return self.__class__(pa_new, po_new)

        return NotImplemented

    def __imul__(self, r) -> "Vecteur3D":
        if hasattr(r, "rot_quat_form"):
            ### rotation of the 2 points
            ### Point3D imul already defined
            self.po *= r
            self.pa *= r
            return self
        elif self._is_scalar_or_1d(r):
            ### here the multiplication afect the norm of the vector
            new_v = self.v * r
            self.pa = self.po + new_v
            return self
        else:
            return NotImplemented

    def __mul__(self, r) -> "Vecteur3D":
        if hasattr(r, "rot_quat_form"):
            ### rotation of the 2 points
            ### Point3D imul already defined
            new_po = self.po * r
            new_pa = self.pa * r
            return self.__class__(new_pa, new_po)
        elif self._is_scalar_or_1d(r):
            ### here the multiplication afect the norm of the vector
            new_v = self.v * r
            new_pa = self.po + new_v
            return self.__class__(new_pa, self.po.copy)

        return NotImplemented

    def __irshift__(self, v):
        """
        Translation of the vector
        moves the 2 points accordingly

        Do not confuse with the sum of vectors
        """
        if isinstance(v, Vecteur3D):
            self.po += v.v
            self.pa += v.v
            return self
        return NotImplemented

    def __add__(self, v: "Vecteur3D") -> "Vecteur3D":
        if isinstance(v, Vecteur3D):
            new_pa = self.pa + v.v
            return Vecteur3D(new_pa, self.po)
        # elif self._is_scalar_or_1d(v): ### should it be implemented ?
        return NotImplemented

    def __iadd__(self, v: "Vecteur3D") -> "Vecteur3D":
        if isinstance(v, Vecteur3D):
            self.pa += v.v
            return self
        return NotImplemented

    def __sub__(self, v: "Vecteur3D") -> "Vecteur3D":
        if isinstance(v, Vecteur3D):
            new_pa = self.pa - v.v
            return self.__class__(new_pa, self.po)
        return NotImplemented

    def __isub__(self, v: "Vecteur3D") -> "Vecteur3D":
        if isinstance(v, Vecteur3D):
            self.pa -= v.v
            return self
        return NotImplemented

    def __matmul__(self, q: "Vecteur3D") -> "Vecteur3D":
        """
        Dot product of vectors
        """
        return (-1) * super().__mul__(q).w

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.v.x.tolist()}, y={self.v.y.tolist()}, z={self.v.z.tolist()})"

    def __getitem__(self, key):
        return self.__class__(self.pa[key], point_origine=self.po[key])


class Rotation:
    def __init__(self, angle: float, vect_dir: Vecteur3D):
        self.angle = np.float64(angle)  ### rad
        self.vect_dir = vect_dir.to_unitaire  ### direction from the local origin

    @property
    def rot_quat_form(self) -> "Quaternion":
        # return Quaternion( np.cos(self.angle/2) , *(np.sin(self.angle/2)*self.vect_dir.to_unitaire.point.T) )
        return Quaternion(
            np.cos(self.angle / 2),
            *(np.sin(self.angle / 2) * self.vect_dir.v.point.T),
        )

    @property
    def copy(self) -> "Rotation":
        return Rotation(self.angle, self.vect_dir)

    # def __mul__(self, q) -> "Point3D":
    #     if isinstance(q, Vecteur3D):
    #         return q.__mul__(self.rot_quat_form)
    #     if isinstance(q, Point3D):
    #         return (
    #             Point3D(
    #                 *self.rot_quat_form.__mul__(q - self.origine)
    #                 .__mul__(self.rot_quat_form.inverse)
    #                 .point.T
    #             )
    #             + self.origine
    #         )

    #     return NotImplemented

    def __imul__(self, r: "Rotation") -> "Vecteur3D":
        ### makes the rotation Rotable
        if isinstance(r, Rotation):
            self.vect_dir *= r
            return self

        return NotImplemented

    # def __irshift__(self, v: "Vecteur3D"):
    #     if isinstance(v, Vecteur3D):
    #         self.origine += v
    #         self.vect_dir += v
    #         return self

    #     return NotImplemented
