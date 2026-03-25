#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
#  Settings
# ============================================================

# .det files to merge (add as many as needed)
DET_FILES = [
    'det/output.det',
    # 'det/run2.det',
]

# Scan grid (must match the .pri generation parameters)
X_RANGE = (-50, 50, 101)   # (min_nm, max_nm, n_pixels)
Y_RANGE = (-100, 100, 201)

# Target / Reference pixel coordinates (x_index, y_index)
# Set to None to disable contrast annotation
TARGET_PX    = (50, 100)    # e.g. center of PMMA line
REFERENCE_PX = (30, 100)    # e.g. silicon region

BOX_SIZE = 3   # NxN averaging box

OUTPUT_FILE = 'images/sem_image.png'

# ============================================================
#  Load & merge .det files
# ============================================================

electron_dtype = np.dtype([
    ('x', '=f'), ('y', '=f'), ('z', '=f'),
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'),
    ('E', '=f'), ('px', '=i'), ('py', '=i'),
])

chunks = []
for f in DET_FILES:
    data = np.fromfile(f, dtype=electron_dtype)
    print(f"  {f}: {len(data)} electrons")
    chunks.append(data)

detected = np.concatenate(chunks)
print(f"Total: {len(detected)} electrons ({len(DET_FILES)} file(s))")

# ============================================================
#  Build image
# ============================================================

xpx = np.linspace(*X_RANGE)
ypx = np.linspace(*Y_RANGE)

image, _, _ = np.histogram2d(
    detected['px'], detected['py'],
    bins=[len(xpx), len(ypx)],
    range=[[0, len(xpx)], [0, len(ypx)]],
)

# ============================================================
#  Plot
# ============================================================

fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(image.T, cmap='gray', origin='lower',
               extent=[xpx[0], xpx[-1], ypx[0], ypx[-1]])
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title('Simulated SEM Image (500 eV)')
fig.colorbar(im, ax=ax, label='Detected electrons')

# ============================================================
#  Contrast annotation
# ============================================================

if TARGET_PX is not None and REFERENCE_PX is not None:
    half = BOX_SIZE // 2

    def box_mean(img, cx, cy):
        x0 = max(cx - half, 0)
        x1 = min(cx + half + 1, img.shape[0])
        y0 = max(cy - half, 0)
        y1 = min(cy + half + 1, img.shape[1])
        return img[x0:x1, y0:y1].mean()

    T = box_mean(image, *TARGET_PX)
    R = box_mean(image, *REFERENCE_PX)
    contrast = (T - R) / R if R != 0 else float('nan')

    print(f"Target  {TARGET_PX}: mean = {T:.1f}")
    print(f"Reference {REFERENCE_PX}: mean = {R:.1f}")
    print(f"Contrast (T-R)/R = {contrast:.4f}")

    # Draw boxes on the image
    dx = xpx[1] - xpx[0]  # nm per pixel
    dy = ypx[1] - ypx[0]

    for (px_x, px_y), color, label in [
        (TARGET_PX, 'red', 'T'),
        (REFERENCE_PX, 'cyan', 'R'),
    ]:
        # Convert pixel index to nm coordinates
        nm_x = xpx[0] + (px_x - half) * dx
        nm_y = ypx[0] + (px_y - half) * dy
        rect = patches.Rectangle(
            (nm_x, nm_y), BOX_SIZE * dx, BOX_SIZE * dy,
            linewidth=1.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(nm_x + BOX_SIZE * dx + dx, nm_y + BOX_SIZE * dy / 2,
                label, color=color, fontsize=10, fontweight='bold',
                va='center')

    ax.text(0.02, 0.02, f'(T\u2212R)/R = {contrast:.4f}',
            transform=ax.transAxes, color='yellow', fontsize=11,
            fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_FILE}")
