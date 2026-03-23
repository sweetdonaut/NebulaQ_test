#!/usr/bin/env python3
import numpy as np

electron_dtype = np.dtype([
    ('x', '=f'), ('y', '=f'), ('z', '=f'),
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'),
    ('E', '=f'),
    ('px', '=i'), ('py', '=i')
])

z = 35            # 起始高度，需在 PMMA 頂部 (30nm) 之上
energy = 500      # 電子束能量 (eV)
epx = 100         # 每像素電子數
sigma = 1         # 光斑標準差 (nm)

xpx = np.linspace(-50, 50, 101)    # 掃描範圍覆蓋 PMMA 線條兩側
ypx = np.linspace(-100, 100, 201)

with open('pri/sem.pri', 'wb') as file:
    for i, xmid in enumerate(xpx):
        for j, ymid in enumerate(ypx):
            N = np.random.poisson(epx)
            buf = np.empty(N, dtype=electron_dtype)
            buf['x'] = np.random.normal(xmid, sigma, N)
            buf['y'] = np.random.normal(ymid, sigma, N)
            buf['z'] = z
            buf['dx'] = 0; buf['dy'] = 0; buf['dz'] = -1
            buf['E'] = energy
            buf['px'] = i; buf['py'] = j
            buf.tofile(file)

print(f"已生成 sem.pri，{len(xpx)}x{len(ypx)} 像素，約 {len(xpx)*len(ypx)*epx/1e6:.1f}M 電子")