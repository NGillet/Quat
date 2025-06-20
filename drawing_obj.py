from abc import ABC, abstractmethod

import numpy as np

from quaternion import Point3D, Rotation, Vecteur3D, eps


class Drawable_object(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ### rotation defined just for this object: they are FIX one to an other !
        self._rotation_propre = []

    @property
    @abstractmethod
    def project_on_screen(self):
        pass

    @abstractmethod
    def __imul__(self, r: "Rotation", update_trasformation_propre: bool = True):
        ### make the rotation_propre move with the object
        ### but only if it NOT 'r' itself a rotation_propre: update_trasformation_propre
        if update_trasformation_propre:
            if isinstance(r, Rotation):
                for rp in self._rotation_propre:
                    rp *= r
                return self
            return NotImplemented
        else:
            pass

    @abstractmethod
    def __irshift__(self):
        pass

    def add_rot(self, r: "Rotation"):
        if isinstance(r, Rotation):
            self._rotation_propre.append(r)

    # def __repr__(self):
    #     return (
    #         f"{self.__class__.__name__}(rotation_propre={len(self._rotation_propre)})"
    #     )


class Groupe_drawable_object:
    def __init__(
        self,
    ):
        self.list_of_obj = []  ### list[]

    def __len__(self):
        return len(self.list_of_obj)

    @property
    def size(self):
        return len(self)

    def add_in(self, obj):
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, Drawable_object):
                    self.list_of_obj.append(o)
                else:
                    print(f" obj {type(o)}  not a Drawable_object")
        else:
            self.add_in([obj])

    def apply_rotation_propre(self):
        pass
        # for obj in self.list_of_obj:
        #     if len( obj._rotation_propre ):
        #         for rp in obj._rotation_propre:
        #             #obj *= rp
        #             obj.__imul__(rp, update_trasformation_propre=False)

    ### for fun: us add_in
    def __or__(self, obj):
        self.list_of_obj.append(obj)
        return self

    def __imul__(self, r: "Rotation") -> "Groupe_drawable_object":
        if isinstance(r, Rotation):
            for obj in self.list_of_obj:
                if len(obj._rotation_propre):
                    print(
                        "grp : ",
                        obj._rotation_propre[0].vect_dir,
                        "| cercle :",
                        obj.vecteur_normal,
                    )
                obj *= r
            return self

        return NotImplemented

    def __irshift__(self, v: "Vecteur3D") -> "Groupe_drawable_object":
        if isinstance(v, Vecteur3D):
            for obj in self.list_of_obj:
                obj >>= v
            return self

        return NotImplemented


class Screen:
    def __init__(
        self,
        focal: float,
        FoV: float = 100.0,
        size_pixel: int = 600,
        DoV: float = 10.0,
    ):
        """
        DoV in unit of f
        """

        assert (FoV >= 90) & (FoV <= 140)
        self.FoV_deg = FoV
        self.demi_FoV_rad = (self.FoV_deg / 2.0) * np.pi / 180.0
        assert focal > 0
        self.focal = focal

        self.demi_screen_size = self.focal * np.tan(
            self.demi_FoV_rad
        )  ### half screen size

        self.size_pixel = size_pixel

        self.DoV = focal * DoV

    @property
    def up(self):
        return self.demi_screen_size

    @property
    def down(self):
        return -self.demi_screen_size

    @property
    def left(self):
        return self.demi_screen_size

    @property
    def right(self):
        return -self.demi_screen_size

    def __repr__(self) -> str:
        return f"Focal : {self.focal},\nFoV : {self.FoV_deg} : {self.demi_screen_size}"


class Point_obj(Point3D, Drawable_object):
    def __init__(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
    ):
        super().__init__(x=x, y=y, z=z)

    def __imul__(self, r) -> "Point_obj":
        super().__imul__(r)
        return self

    def __irshift__(self, v) -> "Point_obj":
        super().__irshift__(v)
        return self

    # def __repr__(self) -> str:
    #     return (
    #         f"Point_obj(x={self.x.tolist()}, y={self.y.tolist()}, z={self.z.tolist()})"
    #     )

    def get_coord_on_screen(self, S: "Screen"):
        f = -S.focal

        if f == -np.inf:
            return self.point[:, 1:]

        denominator = self.point[:, 0] - f  ### shape (N,)
        return (
            self.point[:, 1:] * (-f) / denominator[:, None]
        )  ### shape (N,2) ### nan if divided by zeros

    def project_on_screen(self, S: "Screen"):
        cond_x = self.point[:, 0] >= 0
        dist = (self.point * self.point).sum(axis=1)
        cond_d = dist <= S.DoV**2

        yz_screen = self.get_coord_on_screen(S)  ### shape (N,2)
        ### remove points out of the screen (left,right,up,down)
        cond_ys = np.abs(yz_screen[:, 0]) <= S.demi_screen_size
        cond_zs = np.abs(yz_screen[:, 1]) <= S.demi_screen_size

        ### the whole visible condition
        cond_in_FoV = cond_x & cond_d & cond_ys & cond_zs

        ### nan non visible points
        yz_screen[~cond_in_FoV] = np.nan  ### shape (N,2)

        j = (
            -(yz_screen[:, 0] - S.demi_screen_size)
            / (2 * S.demi_screen_size)
            * S.size_pixel
        )
        i = (
            -(yz_screen[:, 1] - S.demi_screen_size)
            / (2 * S.demi_screen_size)
            * S.size_pixel
        )
        return i, j


class Vecteur_obj(Vecteur3D, Drawable_object):
    def __init__(
        self,
        point_arrivee: "Point3D",
        point_origine: "Point3D" = None,
    ):
        super().__init__(point_arrivee, point_origine)

    def __imul__(self, r) -> "Point_obj":
        super().__imul__(r)
        return self

    def __irshift__(self, v) -> "Point_obj":
        super().__irshift__(v)
        return self

    def segment_enddings_3D(self, S: "Screen"):
        """
        Return segment enddings points coord in 3D
        ordered from closet to screen to further away (-> X positive)
        """

        point_origine = np.copy(self.po.point)  ### shape (N,3)
        point_arrivee = np.copy(self.pa.point)  ### shape (N,3)

        re_order = (
            point_arrivee[:, 0] < point_origine[:, 0]
        )  # shape (N,), True where swap
        # re_order is shape (N,), True where you want to swap the two rows
        tmp = point_origine[re_order].copy()
        point_origine[re_order] = point_arrivee[re_order]
        point_arrivee[re_order] = tmp

        return point_origine, point_arrivee  ### 2x shape (N,3)

    def intersection_with_screen_3D(self, S: "Screen"):
        """
        Return screen intersection point coord in 3D
            - the 'screen' is the plane x=0
            - there is One intersection MAX
            - the intersection exist only if Vx!=0 (// vector), nan if //
        """
        A = self.po.point  ### np.ndarray (N,3,)
        V = self.v.point  ### np.ndarray (N,3,)
        V = V * np.sign(V[:, 0])[:, None]  ### Vx toward positive X

        x_zero = np.zeros(self.shape)  ### shape (N,)
        y_zero = np.full(self.shape, np.nan)  ### shape (N,)
        z_zero = np.full(self.shape, np.nan)  ### shape (N,)

        ok = np.abs(V[:, 0]) > eps
        idx_ok = np.nonzero(ok)[0]
        if idx_ok.size:
            y_zero[idx_ok] = A[idx_ok, 1] - A[idx_ok, 0] * V[idx_ok, 1] / V[idx_ok, 0]
            z_zero[idx_ok] = A[idx_ok, 2] - A[idx_ok, 0] * V[idx_ok, 2] / V[idx_ok, 0]

        return np.array([x_zero, y_zero, z_zero]).T  ### 1x shape (N,3)

    def intersection_with_sphere_3D(self, S: "Screen"):
        """
        Return sphere intersection points coord in 3D
            - intersections of the STRAIGHT (define by this vector) with the sphere (Screen Distance of View)
            - there is 0, 1 or 2 intersections
            - ordered from closet to screen to further away (-> X positive)
            - nan if no intersection
        """

        A = self.po.point  ### np.ndarray (N,3,)
        V = self.v.point  ### np.ndarray (N,3,)
        V = V * np.sign(V[:, 0])[:, None]  ### Vx toward positive X

        f = -S.focal

        ### get the point at the minimum distance from the origine
        t_Amin = -np.vecdot(A, V)  ### np.ndarray (N,) Nchanel
        Amin = A + (V * t_Amin[:, None])  ### np.ndarray (N,3,)

        if S.DoV < np.inf:
            d_Amin = (Amin**2).sum(axis=1)
            ### get the 2 intersection points
            tpos = np.sqrt(
                S.DoV**2 - d_Amin
            )  ### np.ndarray (N,) => nan when the point is too far
            # tneg = -np.sqrt( S.DoV**2 - d_Amin ) ### np.ndarray (N,)

            inter_pos = Amin + (V.T * tpos).T  ### np.ndarray (N,3,)
            inter_neg = Amin + (V.T * (-tpos)).T  ### np.ndarray (N,3,)

            return inter_pos, inter_neg  ### 2x shape (N,3)

        else:
            xs_pos = np.full_like(t_Amin, np.inf)  ### np.ndarray (N,) Nchanel
            ys_pos = np.full_like(t_Amin, np.nan)
            zs_pos = np.full_like(t_Amin, np.nan)
            xs_neg = np.zeros(self.shape)
            ys_neg = np.full_like(t_Amin, np.nan)
            zs_neg = np.full_like(t_Amin, np.nan)

            ok_Vx = np.abs(V[:, 0]) > eps
            ### positive intersection at infinity
            ys_pos[ok_Vx] = (V[ok_Vx, 1] / V[ok_Vx, 0]) * (-f)
            zs_pos[ok_Vx] = (V[ok_Vx, 2] / V[ok_Vx, 0]) * (-f)

            ### negative intersection replace by the x=0 plane intersection
            ys_neg[ok_Vx] = A[ok_Vx, 1] - A[ok_Vx, 0] * V[ok_Vx, 1] / V[ok_Vx, 0]
            zs_neg[ok_Vx] = A[ok_Vx, 2] - A[ok_Vx, 0] * V[ok_Vx, 2] / V[ok_Vx, 0]

            p_pos = np.array([xs_pos[0], ys_pos[0], zs_pos[0]]).T  ### shape (N,3)
            p_neg = np.array([xs_neg[1], ys_neg[1], zs_neg[1]]).T  ### shape (N,3)

            return p_pos, p_neg  ### 2x shape (N,3)

    def intersection_with_FoV_3D(self, S: "Screen"):
        """
        Return field of view (Screen borders) intersection points coord in 3D
            - intersections of the STRAIGHT (define by this vector) with the field of view (Screen borders)
            - there is 0, 1 or 2 intersections
            - ordered from FURTHER away to screen to closest away (infinity point first)
            - nan if no intersection
        """

        A = self.po.point  ### np.ndarray (N,3,)
        Ax, Ay, Az = A.T  ### np.ndarray (N,)
        V = self.v.point  ### np.ndarray (N,3,)
        V = V * np.sign(V[:, 0])[:, None]  ### Vx toward positive X
        Vx, Vy, Vz = V.T  ### np.ndarray (N,)

        f = -S.focal
        y_size = S.demi_screen_size
        z_size = S.demi_screen_size  ### TODO : rectangle screen

        M = self.shape

        ### Prepare arrays to hold up to two (y,z) hits per ray : on screen coordinates!
        # We’ll store row=0 for the first hit, row=1 for the second hit.
        y_sub = np.full((2, M), np.nan, dtype=float)  ### np.ndarray (2, N,)
        z_sub = np.full((2, M), np.nan, dtype=float)
        x_sub = np.full((2, M), np.nan, dtype=float)  ### for ordering points at the end
        hits = np.zeros(M, dtype=int)  ### np.ndarray (N,)

        def get_coord(ts, ca, xa_, Vx_, Vc_, f_):
            """
            Generic formula of projection of y_3d (or z_3d) on screen => y_s (z_s)
            c being y or z coordinate
            """
            return -f_ * (Vc_ * ts + ca) / (-f_ + Vx_ * ts + xa_)

        def get_t(cs, ca, xa_, Vx_, Vc_, f_):
            """
            Get param 't' to be on screen
            cs being y or z on screen (xs=0)
            """
            return (-f_ * ca + f_ * cs - cs * xa_) / (cs * Vx_ + f_ * Vc_)

        ### 1 : Intersection left y=+y_side:
        # valid_1 = np.abs(Vy) > eps
        # if valid_1.any():
        t1 = get_t(y_size, Ay, Ax, Vx, Vy, f)
        zs1 = get_coord(t1, Az, Ax, Vx, Vz, f)
        x3d1 = Ax + Vx * t1
        y3d1 = Ay + Vy * t1
        z3d1 = Az + Vz * t1
        # flag_in_FoV = valid_1 & (abs(zs1) <= z_size) & (x3d1>0)
        flag_in_FoV = (abs(zs1) <= z_size) & (x3d1 > 0)
        if flag_in_FoV.any():
            z_sub[hits[flag_in_FoV], flag_in_FoV] = z3d1[flag_in_FoV]
            y_sub[hits[flag_in_FoV], flag_in_FoV] = y3d1[flag_in_FoV]
            x_sub[hits[flag_in_FoV], flag_in_FoV] = x3d1[flag_in_FoV]
            hits[flag_in_FoV] += 1

        ### 2 : Intersection right y=-y_side:
        # valid_2 = np.abs(Vy) > eps
        # if valid_2.any():
        t2 = get_t(-y_size, Ay, Ax, Vx, Vy, f)
        x3d2 = Ax + Vx * t2
        y3d2 = Ay + Vy * t2
        z3d2 = Az + Vz * t2
        zs2 = get_coord(t2, Az, Ax, Vx, Vz, f)
        # flag_in_FoV = valid_2 & (abs(zs2) <= z_size) & (x3d2>0)
        flag_in_FoV = (abs(zs2) <= z_size) & (x3d2 > 0)
        if flag_in_FoV.any():
            z_sub[hits[flag_in_FoV], flag_in_FoV] = z3d2[flag_in_FoV]
            y_sub[hits[flag_in_FoV], flag_in_FoV] = y3d2[flag_in_FoV]
            x_sub[hits[flag_in_FoV], flag_in_FoV] = x3d2[flag_in_FoV]
            hits[flag_in_FoV] += 1

        ### 3 : Intersection top z=+z_side:
        # valid_3 = np.abs(Vz) > eps
        # if valid_3.any():
        t3 = get_t(z_size, Az, Ax, Vx, Vz, f)
        ys3 = get_coord(t3, Ay, Ax, Vx, Vy, f)
        x3d3 = Ax + Vx * t3
        y3d3 = Ay + Vy * t3
        z3d3 = Az + Vz * t3
        # flag_in_FoV = valid_3 & (np.abs(ys3) <= y_size) & (x3d3>0)
        flag_in_FoV = (np.abs(ys3) <= y_size) & (x3d3 > 0)
        if flag_in_FoV.any():
            z_sub[hits[flag_in_FoV], flag_in_FoV] = z3d3[flag_in_FoV]
            y_sub[hits[flag_in_FoV], flag_in_FoV] = y3d3[flag_in_FoV]
            x_sub[hits[flag_in_FoV], flag_in_FoV] = x3d3[flag_in_FoV]
            hits[flag_in_FoV] += 1

        ### 4 : Intersection bottom z=-z_side:
        # valid_4 = np.abs(Vz) > eps
        # if valid_4.any():
        t4 = get_t(-z_size, Az, Ax, Vx, Vz, f)
        ys4 = get_coord(t4, Ay, Ax, Vx, Vy, f)
        x3d4 = Ax + Vx * t4
        y3d4 = Ay + Vy * t4
        z3d4 = Az + Vz * t4
        # flag_in_FoV = valid_4 & (np.abs(ys4) <= y_size) & (x3d4>0)
        flag_in_FoV = (np.abs(ys4) <= y_size) & (x3d4 > 0)
        if flag_in_FoV.any():
            z_sub[hits[flag_in_FoV], flag_in_FoV] = z3d4[flag_in_FoV]
            y_sub[hits[flag_in_FoV], flag_in_FoV] = y3d4[flag_in_FoV]
            x_sub[hits[flag_in_FoV], flag_in_FoV] = x3d4[flag_in_FoV]
            hits[flag_in_FoV] += 1
        ### at this point there should be only 2 hits max ! (0 and 1 possibly)

        ### I want the further away point of the screen first (like inf!)
        re_order = x_sub[0] < x_sub[1]
        # re_order is shape (N,), True where you want to swap the two rows
        mask = re_order[None, :]  # shape (1,N) → broadcasts over 2 rows

        # x_sub, y_sub and z_sub are each shape (2,N)
        x_sub = np.where(mask, x_sub[::-1], x_sub)
        y_sub = np.where(mask, y_sub[::-1], y_sub)
        z_sub = np.where(mask, z_sub[::-1], z_sub)

        p_int1 = np.array([x_sub[0], y_sub[0], z_sub[0]]).T  ### shape (N,3)
        p_int2 = np.array([x_sub[1], y_sub[1], z_sub[1]]).T  ### shape (N,3)

        return p_int1, p_int2  ### 2x shape (N,3) ### nan if points are not defines

    def get_coord_on_screen_px(self, p, S: "Screen"):
        y_size = S.demi_screen_size
        z_size = S.demi_screen_size

        ys, zs = self.get_coord_on_screen(p, S).T
        i = -(zs - z_size) / (2 * z_size) * S.size_pixel
        j = -(ys - y_size) / (2 * y_size) * S.size_pixel

        return i, j  ### 2x shape (N,)

    def get_coord_on_screen(self, p, S: "Screen"):
        f = -S.focal
        denominator = p[:, 0] - f  # shape (N,)
        return p[:, 1:] * (-f) / denominator[:, None]  ### shape (N,2)

    def flag_in_field_of_view(self, p, S: "Screen"):
        y_size = S.demi_screen_size
        z_size = S.demi_screen_size  ### TODO : rectangle screen

        ys, zs = self.get_coord_on_screen(p, S).T
        dp2 = (p * p).sum(axis=1)

        return (
            (abs(ys) <= y_size + eps)
            & (abs(zs) <= z_size + eps)
            & (dp2 < S.DoV**2)
            & (p[:, 0] >= -eps)
        )

    def project_on_screen(
        self,
        S: "Screen",
    ):
        """
        A vector is a segment to draw : it has 2 extremity, O (origine) A (arrivee)
        For the drawing Ax > Ox so the vector Vx>0
        """
        ### get all possible points 3D coords
        ### get if points are in FoV
        p_o, p_a = self.segment_enddings_3D(S)
        o_inside = self.flag_in_field_of_view(p_o, S)
        a_inside = self.flag_in_field_of_view(p_a, S)
        p_infi, p_neg = self.intersection_with_sphere_3D(S)
        infi_inside = self.flag_in_field_of_view(p_infi, S)
        neg_inside = self.flag_in_field_of_view(p_neg, S)
        p_int1, p_int2 = self.intersection_with_FoV_3D(S)
        int1_inside = self.flag_in_field_of_view(p_int1, S)
        int2_inside = self.flag_in_field_of_view(p_int2, S)

        p_zero = self.intersection_with_screen_3D(S)
        zero_inside = self.flag_in_field_of_view(p_zero, S)
        ### get all point px coords
        i_o_px, j_o_px = self.get_coord_on_screen_px(p_o, S)
        i_a_px, j_a_px = self.get_coord_on_screen_px(p_a, S)
        i_infi_px, j_infi_px = self.get_coord_on_screen_px(p_infi, S)
        i_neg_px, j_neg_px = self.get_coord_on_screen_px(p_neg, S)
        i_int1_px, j_int1_px = self.get_coord_on_screen_px(p_int1, S)
        i_int2_px, j_int2_px = self.get_coord_on_screen_px(p_int2, S)

        i_zero_px, j_zero_px = self.get_coord_on_screen_px(p_zero, S)

        ### All outputs start as NaN
        i_pts = np.full((2, self.shape), np.nan, dtype=float)
        j_pts = np.full((2, self.shape), np.nan, dtype=float)

        ### 1) BOTH in (Origin and Arrivee are on‐screen):
        mask_both = o_inside & a_inside
        if mask_both.any():
            # row 0 = infinity‐hit
            i_pts[0, mask_both] = i_o_px[mask_both]
            j_pts[0, mask_both] = j_o_px[mask_both]
            i_pts[1, mask_both] = i_a_px[mask_both]
            j_pts[1, mask_both] = j_a_px[mask_both]

        ### 2) Origin in: ### the second is inf or int1 or int2
        mask_O = o_inside & ~a_inside
        if mask_O.any():
            i_pts[0, mask_O] = i_o_px[mask_O]
            j_pts[0, mask_O] = j_o_px[mask_O]
            ### 1) can be the inf point
            mask = mask_O & infi_inside
            if mask.any():
                i_pts[1, mask] = i_infi_px[mask]
                j_pts[1, mask] = j_infi_px[mask]
            ### 2) if not inf then it is int1 or int2
            mask = mask_O & ~infi_inside & int1_inside
            if mask.any():
                i_pts[1, mask] = i_int1_px[mask]
                j_pts[1, mask] = j_int1_px[mask]
            mask = mask_O & ~infi_inside & ~int1_inside & int2_inside
            if mask.any():
                i_pts[1, mask] = i_int2_px[mask]
                j_pts[1, mask] = j_int2_px[mask]
            ### 3) there should not be other case

        ### 3) Arrivee in: ### the second is zero or int1
        mask_A = ~o_inside & a_inside
        if mask_A.any():
            i_pts[0, mask_A] = i_a_px[mask_A]
            j_pts[0, mask_A] = j_a_px[mask_A]

            ### 1) can be the zero point
            mask = mask_A & zero_inside
            if mask.any():
                i_pts[1, mask] = i_zero_px[mask]
                j_pts[1, mask] = j_zero_px[mask]
            ### 2) if not inf then it is int2 (can't be int1)
            mask = mask_A & ~zero_inside & int2_inside
            if mask.any():
                i_pts[1, mask] = i_int2_px[mask]
                j_pts[1, mask] = j_int2_px[mask]
            mask = mask_A & ~zero_inside & ~int2_inside & int1_inside
            if mask.any():
                i_pts[1, mask] = i_int1_px[mask]
                j_pts[1, mask] = j_int1_px[mask]
            ### 3) there should not be other case

        ### 4) None of O and A are in: but the segment can still intersect the FoV
        mask_None = ~o_inside & ~a_inside
        if mask_None.any():
            ### 4.1) can be the inf point
            mask = mask_None & infi_inside
            if mask.any():
                ### check if infi is inside [OA]
                ### [OA]
                v_i = i_a_px - i_o_px  # shape (N,)
                v_j = j_a_px - j_o_px
                ### [O-int1]
                u_i = i_infi_px - i_o_px  # shape (N,)
                u_j = j_infi_px - j_o_px
                dot = u_i * (u_i - v_i) + u_j * (u_j - v_j)  # shape (N,)
                between = dot <= 0

                mask_4 = mask & between
                if mask_4.any():
                    i_pts[0, mask_4] = i_infi_px[mask]
                    j_pts[0, mask_4] = j_infi_px[mask]
                    ### can be zero or int1
                    mask_41 = mask & neg_inside
                    if mask_41.any():
                        i_pts[1, mask_41] = i_neg_px[mask_4]
                        j_pts[1, mask_41] = j_neg_px[mask_4]
                    mask_41 = mask & ~neg_inside & int2_inside
                    if mask_41.any():
                        i_pts[1, mask_41] = i_int2_px[mask_4]
                        j_pts[1, mask_41] = j_int2_px[mask_4]

            ### 4.2) inf is not in : then take the two intersections
            mask = mask_None & ~infi_inside & int1_inside & int2_inside
            if mask.any():
                ### check if int1 is inside [OA]
                ### [OA]
                v_i = i_a_px - i_o_px  # shape (N,)
                v_j = j_a_px - j_o_px
                ### [O-int1]
                u_i = i_int1_px - i_o_px  # shape (N,)
                u_j = j_int1_px - j_o_px
                dot = u_i * (u_i - v_i) + u_j * (u_j - v_j)  # shape (N,)
                between = dot <= 0

                mask_42 = mask & between
                if mask_42.any():
                    i_pts[0, mask_42] = i_int1_px[mask_42]
                    j_pts[0, mask_42] = j_int1_px[mask_42]
                    i_pts[1, mask_42] = i_int2_px[mask_42]
                    j_pts[1, mask_42] = j_int2_px[mask_42]

        ### final, remove nan
        mask_draw = ~np.isnan(i_pts) & ~np.isnan(j_pts)  ### shape (2, N)
        mask_draw = mask_draw.all(axis=0)  ### shape (N,)
        return i_pts[:, mask_draw], j_pts[:, mask_draw]
