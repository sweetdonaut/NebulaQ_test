#!/usr/bin/env python3
"""Generate primary electron beam using HourglassGaussianBeam for PMMA sample.

Based on sem-pri.py, replacing the parallel Gaussian beam with
JMONSEL-style hourglass beam (convergent cone + Gaussian spot at focus).
"""
import numpy as np
from hourglass_beam import HourglassGaussianBeam

# SEM parameters
z_focus = 30          # focal plane = PMMA top surface (nm)
energy = 500          # beam energy (eV)
epx = 100             # average electrons per pixel
sigma = 1             # Gaussian spot sigma (nm)
aperture = 0.015      # convergence half-angle (rad), 15 mrad
offset = 5            # distance above focus to start electrons (nm)

# Scan range (same as sem-pri.py)
xpx = np.linspace(-50, 50, 101)
ypx = np.linspace(-100, 100, 201)

# Create hourglass beam
gun = HourglassGaussianBeam(width=sigma, center=[0, 0, z_focus])
gun.beam_energy = energy
gun.beam_direction = [0, 0, -1]
gun.angular_aperture = aperture
gun.offset = offset

with open('pri/hourglass_sem.pri', 'wb') as file:
    for i, xmid in enumerate(xpx):
        for j, ymid in enumerate(ypx):
            N = np.random.poisson(epx)
            gun.center = [xmid, ymid, z_focus]
            buf = gun.create_pri_buffer(N, px=i, py=j)
            buf.tofile(file)

n_total = len(xpx) * len(ypx) * epx
print(f"Generated hourglass_sem.pri")
print(f"  {len(xpx)}x{len(ypx)} pixels, ~{n_total/1e6:.1f}M electrons")
print(f"  Energy: {energy} eV, sigma: {sigma} nm, aperture: {aperture*1000:.0f} mrad")
print(f"  Focus plane: z={z_focus} nm, start: z~{z_focus + offset} nm")
