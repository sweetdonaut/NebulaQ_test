#!/usr/bin/env python3
"""Faithful Python port of JMONSEL HourglassGaussianBeam.

Original Java source:
  gov.nist.nanoscalemetrology.JMONSEL.HourglassGaussianBeam
  Author: John Villarrubia, NIST, March 26, 2012
  https://github.com/usnistgov/EPQ/blob/master/src/gov/nist/
    nanoscalemetrology/JMONSEL/HourglassGaussianBeam.java

Supporting classes referenced (Electron.java):
  - Electron(pos, theta, phi, energy)   — constructor
  - updateDirection(dTheta, dPhi)       — rotate direction
  - candidatePoint(dS)                  — move along direction
  https://github.com/usnistgov/EPQ/blob/master/src/gov/nist/
    microanalysis/NISTMonte/Electron.java

This port preserves the original algorithm exactly, adapted from
single-electron Java to numpy-vectorised Python.  The math is
unit-agnostic: all length parameters (width, center, offset) must
use the same unit (e.g. nm for Nebula).  Energy is passed through
as-is (e.g. eV for Nebula).
"""

import numpy as np

# Nebula .pri / .det electron record format
electron_dtype = np.dtype([
    ('x',  '=f'), ('y',  '=f'), ('z',  '=f'),   # position
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'),   # direction
    ('E',  '=f'),                                 # energy
    ('px', '=i'), ('py', '=i'),                   # pixel index
])


class HourglassGaussianBeam:
    """Hourglass-shaped beam with Gaussian distribution at best focus.

    At best focus the probability that an electron lands between
    distance r and r+dr from centre is  r·exp[-r²/(2σ²)]·dr,
    where σ = width.

    Electrons converge above focus and diverge below it.  Directions
    are uniformly distributed in solid angle within a cone of
    half-angle = angular_aperture around the mean beam direction.

    Parameters
    ----------
    width : float
        Gaussian σ at best focus.  Each transverse coordinate has
        std dev = width.  RMS radius = √2·width.
        Relation to d56 convention: width = 0.3902·d56.
    center : array-like, shape (3,)
        Position of best focus [x, y, z].
    """

    def __init__(self, width, center):
        self._width = float(width)
        self._center = np.asarray(center, dtype=np.float64)
        self._beam_energy = 0.0

        # Java field defaults for beamDirection = [0, 0, 1]
        self._beam_direction = np.array([0., 0., 1.])
        self._mean_theta = 0.0          # acos(1)
        self._mean_phi = 0.0            # atan2(0, 0)
        # Orthonormal basis in focal plane — matches Java init values
        self._x = np.array([0., 1., 0.])
        self._y = np.array([-1., 0., 0.])

        self._angular_aperture = 0.0
        self._cos_aperture = 1.0
        # Java default is 0.01 (SI metres = 1 cm).  This port is
        # unit-agnostic, so there is no sensible default — the caller
        # MUST set offset in whatever length unit the rest of the
        # parameters use (e.g. nm for Nebula).
        self._offset = None

    # ----------------------------------------------------------------
    #  Properties  (mirror Java getters / setters)
    # ----------------------------------------------------------------

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, v):
        self._width = float(v)

    @property
    def beam_energy(self):
        return self._beam_energy

    @beam_energy.setter
    def beam_energy(self, v):
        self._beam_energy = float(v)

    @property
    def center(self):
        return self._center.copy()

    @center.setter
    def center(self, v):
        self._center = np.asarray(v, dtype=np.float64)

    @property
    def beam_direction(self):
        return self._beam_direction.copy()

    @beam_direction.setter
    def beam_direction(self, bD):
        """Java: setBeamDirection — normalise & recompute basis."""
        bD = np.asarray(bD, dtype=np.float64)
        self._beam_direction = bD / np.linalg.norm(bD)
        self._mean_theta = np.arccos(
            np.clip(self._beam_direction[2], -1.0, 1.0))
        self._mean_phi = np.arctan2(
            self._beam_direction[1], self._beam_direction[0])

        # Java: find component with smallest |value|, build x, y
        min_index = int(np.argmin(np.abs(self._beam_direction)))
        temp = np.zeros(3)
        temp[min_index] = 1.0
        self._x = np.cross(self._beam_direction, temp)
        self._x /= np.linalg.norm(self._x)
        self._y = np.cross(self._beam_direction, self._x)

    @property
    def angular_aperture(self):
        return self._angular_aperture

    @angular_aperture.setter
    def angular_aperture(self, v):
        """Java: setAngularAperture — half-angle of cone (radians)."""
        self._angular_aperture = float(v)
        self._cos_aperture = np.cos(self._angular_aperture)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, v):
        self._offset = float(v)

    # ----------------------------------------------------------------
    #  Core algorithm
    # ----------------------------------------------------------------

    def create_electron(self):
        """Single-electron version of Java createElectron().

        Returns
        -------
        position  : ndarray (3,)   — start position (upstream of focus)
        direction : ndarray (3,)   — unit direction vector
        energy    : float
        """
        # -- Step 1  direction within convergence cone ----------------
        # Java: uniform in solid angle on spherical cap
        #   costheta = (1-r) + r*cosAperture
        r = np.random.random()
        cos_theta = (1.0 - r) + r * self._cos_aperture
        theta = np.arccos(cos_theta)
        phi = 2.0 * np.pi * np.random.random()

        # -- Step 2  position at focal plane  (Box-Muller / Rayleigh) -
        # Java: rad = sqrt(-2*log(U)) * mWidth
        u = max(np.random.random(), np.finfo(float).tiny)  # guard log(0)
        rad = np.sqrt(-2.0 * np.log(u)) * self._width
        th = 2.0 * np.pi * np.random.random()
        pos = (self._center
               + rad * np.cos(th) * self._x
               + rad * np.sin(th) * self._y)

        # -- Steps 3-4  Electron(pos, meanTheta, meanPhi, E)
        #               .updateDirection(theta, phi) -----------------
        ct = np.cos(self._mean_theta)
        st = np.sin(self._mean_theta)
        cp = np.cos(self._mean_phi)
        sp = np.sin(self._mean_phi)
        ca = np.cos(theta)
        sa = np.sin(theta)
        cb = np.cos(phi)

        xx = cb * ct * sa + ca * st
        yy = sa * np.sin(phi)
        dx = cp * xx - sp * yy
        dy = cp * yy + sp * xx
        dz = ca * ct - cb * sa * st

        direction = np.array([dx, dy, dz])

        # -- Step 5  candidatePoint(-offset) --------------------------
        if self._offset is None:
            raise ValueError(
                "offset not set — call gun.offset = <value> first "
                "(same length unit as width/center)")
        start = pos - self._offset * direction

        return start, direction, self._beam_energy

    def create_electrons(self, n):
        """Vectorised batch — same math as create_electron(), n times.

        Parameters
        ----------
        n : int
            Number of electrons to generate.

        Returns
        -------
        positions  : ndarray (n, 3)  — start positions
        directions : ndarray (n, 3)  — unit direction vectors
        energy     : float           — beam energy
        """
        if n <= 0:
            return np.empty((0, 3)), np.empty((0, 3)), self._beam_energy

        # -- Step 1  directions in convergence cone -------------------
        # Java: costheta = (1-r) + r*cosAperture
        r = np.random.random(n)
        cos_theta = (1.0 - r) + r * self._cos_aperture
        theta = np.arccos(cos_theta)
        phi = 2.0 * np.pi * np.random.random(n)

        # -- Step 2  positions at focal plane -------------------------
        # Java: rad = sqrt(-2*log(U)) * mWidth   (Rayleigh)
        u = np.random.random(n)
        u = np.maximum(u, np.finfo(float).tiny)   # guard log(0)
        rad = np.sqrt(-2.0 * np.log(u)) * self._width
        th = 2.0 * np.pi * np.random.random(n)
        cos_th = np.cos(th)
        sin_th = np.sin(th)

        # pos = center + rad·cos(th)·x + rad·sin(th)·y
        pos = (self._center[np.newaxis, :]
               + (rad * cos_th)[:, np.newaxis] * self._x[np.newaxis, :]
               + (rad * sin_th)[:, np.newaxis] * self._y[np.newaxis, :])

        # -- Steps 3-4  updateDirection(theta, phi) -------------------
        ct = np.cos(self._mean_theta)
        st = np.sin(self._mean_theta)
        cp = np.cos(self._mean_phi)
        sp = np.sin(self._mean_phi)
        ca = np.cos(theta)
        sa = np.sin(theta)
        cb = np.cos(phi)

        xx = cb * ct * sa + ca * st
        yy = sa * np.sin(phi)
        dx = cp * xx - sp * yy
        dy = cp * yy + sp * xx
        dz = ca * ct - cb * sa * st

        directions = np.column_stack([dx, dy, dz])

        # -- Step 5  candidatePoint(-offset) --------------------------
        if self._offset is None:
            raise ValueError(
                "offset not set — call gun.offset = <value> first "
                "(same length unit as width/center)")
        positions = pos - self._offset * directions

        return positions, directions, self._beam_energy

    def create_pri_buffer(self, n, px=0, py=0):
        """Create n electrons as a Nebula .pri structured array.

        Parameters
        ----------
        n  : int   — number of electrons
        px : int   — pixel x index
        py : int   — pixel y index

        Returns
        -------
        buf : ndarray with dtype = electron_dtype
        """
        positions, directions, _ = self.create_electrons(n)
        buf = np.empty(n, dtype=electron_dtype)
        buf['x']  = positions[:, 0]
        buf['y']  = positions[:, 1]
        buf['z']  = positions[:, 2]
        buf['dx'] = directions[:, 0]
        buf['dy'] = directions[:, 1]
        buf['dz'] = directions[:, 2]
        buf['E']  = self._beam_energy
        buf['px'] = px
        buf['py'] = py
        return buf
