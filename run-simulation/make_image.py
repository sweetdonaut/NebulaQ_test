#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

electron_dtype = np.dtype([
    ('x', '=f'), ('y', '=f'), ('z', '=f'),
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'),
    ('E', '=f'), ('px', '=i'), ('py', '=i')
])

detected = np.fromfile('det/output.det', dtype=electron_dtype)
print(f"偵測到 {len(detected)} 顆電子")

xpx = np.linspace(-50, 50, 101)
ypx = np.linspace(-100, 100, 201)

image, _, _ = np.histogram2d(
    detected['px'], detected['py'],
    bins=[len(xpx), len(ypx)],
    range=[[0, len(xpx)], [0, len(ypx)]]
)

plt.figure(figsize=(6, 10))
plt.imshow(image.T, cmap='gray', origin='lower',
           extent=[xpx[0], xpx[-1], ypx[0], ypx[-1]])
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.title('Simulated SEM Image (500 eV)')
plt.colorbar(label='Detected electrons')
plt.savefig('images/sem_image.png', dpi=150, bbox_inches='tight')
print("影像已儲存至 images/sem_image.png")