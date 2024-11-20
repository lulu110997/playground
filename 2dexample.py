import cvxpy as cp

class min_dist_2D:
    """
    Class for calculating the minimum distance between two ellipsoid type shapes
    """
    def __init__(self, cxa, cya, cxb, cyb, aa, ba, ab, bb):
        self.cxa, self.cya, self.cxb, self.cyb = [cxa, cya, cxb, cyb]

        self.aa, self.ba, self.ab, self.bb = [aa, ba, ab, bb]

    def __setup_problem(self):
        # Decision variables
        xa = cp.Variable((2,1))  # Point on SQ a
        xb = cp.Variable((2,1))  # Point on SQ b

        # Params of SQ a
        self.cxa = cp.Parameter((2, 1), pos=True)  # Centre
        aa = cp.Parameter((2, 1), pos=True)  # Radii

        # Params of SQ b
        self.cxb = cp.Parameter((2, 1), pos=True)  # Centre
        ab = cp.Parameter((2, 1), pos=True)  # Radii

    def get_min_dist(self):
        """
        Solves for the minimum distance
        Returns: minimum distance
        """
        pass

    def set_params(self, cxa, cya, cxb, cyb):
        """
        Set the new centre of the shapes
        """
        pass

obj = min_dist_2D(0, 0, 1, 1, 1, 2, 2, 1)
obj.setup_dpp()