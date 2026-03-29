"""
Three paper figures using a synthetic 3-cluster / 3-feature dataset.
Fig 1 — per-point gradient vectors
Fig 2 — grid interpolation (velocity field)
Fig 3 — single-feature field with mask
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow
from scipy.interpolate import griddata

sys.path.insert(0, 'src')
from featurewind.core import tangent_map as TM, dim_reader as DimReader
from featurewind.physics.grid_computation import build_dilated_support_mask
import featurewind.config as config

# ── 0. Synthetic dataset ────────────────────────────────────────────────────
rng = np.random.default_rng(7)
N_per = 10

def cluster(center, n, noise=0.06):
    return np.clip(rng.normal(center, noise, (n, 3)), 0, 1)

X = np.vstack([
    cluster([0.88, 0.10, 0.10], N_per),  # A — feature_0 dominant
    cluster([0.10, 0.88, 0.10], N_per),  # B — feature_1 dominant
    cluster([0.10, 0.10, 0.88], N_per),  # C — feature_2 dominant
])
labels    = np.array(['A']*N_per + ['B']*N_per + ['C']*N_per)
col_names = ['feature_0', 'feature_1', 'feature_2']

CMAP = {'A': '#D64045', 'B': '#3A7BBF', 'C': '#3AA76D'}
FEAT_C = ['#D64045', '#3A7BBF', '#3AA76D']

# ── 1. Compute embedding + Jacobians ────────────────────────────────────────
print("Computing t-SNE + Jacobians …")
params = ['balanced', '20', '7']
data   = TM.compute_tangent_map_data(X.tolist(), DimReader.tsne, params=params)
pos    = data['positions']                      # (N, 2)
grads  = data['grads'].transpose(0, 2, 1)       # (N, F, 2)
print(f"  pos={pos.shape}, grads={grads.shape}")

# ── 2. Build grid ─────────────────────────────────────────────────────────
GRID_RES = 14
config.set_bounding_box(pos)
xmin, xmax = config.bounding_box[:2]
ymin, ymax = config.bounding_box[2:]
pad = (xmax - xmin) * 0.06
xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

cx = np.linspace(xmin + (xmax-xmin)/(2*GRID_RES), xmax - (xmax-xmin)/(2*GRID_RES), GRID_RES)
cy = np.linspace(ymin + (ymax-ymin)/(2*GRID_RES), ymax - (ymax-ymin)/(2*GRID_RES), GRID_RES)
gx, gy = np.meshgrid(cx, cy)

grid_u = np.zeros((3, GRID_RES, GRID_RES))
grid_v = np.zeros((3, GRID_RES, GRID_RES))
for f in range(3):
    vu = griddata(pos, grads[:, f, 0], (gx, gy), method='linear', fill_value=np.nan)
    vv = griddata(pos, grads[:, f, 1], (gx, gy), method='linear', fill_value=np.nan)
    nn_u = griddata(pos, grads[:, f, 0], (gx, gy), method='nearest')
    nn_v = griddata(pos, grads[:, f, 1], (gx, gy), method='nearest')
    grid_u[f] = np.where(np.isnan(vu), nn_u, vu)
    grid_v[f] = np.where(np.isnan(vv), nn_v, vv)

_, unmasked, _ = build_dilated_support_mask(pos, GRID_RES, 1, (xmin, xmax, ymin, ymax))

# ── helpers ─────────────────────────────────────────────────────────────────
EPS = 1e-9
def normalize(u, v):
    m = np.sqrt(u**2 + v**2) + EPS
    return u/m, v/m

def ax_base(ax, title, fi):
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_facecolor('#f7f7f7')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6, color='#222')

def scatter_pts(ax, s=60):
    for cls in ['A','B','C']:
        m = labels == cls
        ax.scatter(pos[m,0], pos[m,1], c=CMAP[cls], s=s, zorder=5,
                   linewidths=0.5, edgecolors='#222', alpha=0.9)

arrow_len = (xmax - xmin) * 0.18   # uniform arrow length for Fig 1

# ─── Figure 1: per-point gradient vectors ───────────────────────────────────
print("Fig 1 …")
fig1, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
fig1.patch.set_facecolor('white')

for fi, ax in enumerate(axes):
    ax_base(ax, col_names[fi], fi)
    scatter_pts(ax, s=24)
    fc = FEAT_C[fi]
    idx = np.arange(len(pos))
    u = grads[idx, fi, 0];  v = grads[idx, fi, 1]
    un, vn = normalize(u, v)
    dx = un * arrow_len;  dy = vn * arrow_len
    ax.quiver(pos[idx,0], pos[idx,1], dx, dy,
              color=fc, scale=1, scale_units='xy',
              width=0.005, headwidth=4.5, headlength=5,
              headaxislength=4, alpha=0.92, zorder=6)

# legend
from matplotlib.lines import Line2D
hdl = [Line2D([0],[0], marker='o', color='w', markerfacecolor=CMAP[k],
              markersize=9, label=f'Cluster {k}', markeredgecolor='#333',
              markeredgewidth=0.5) for k in ['A','B','C']]
fig1.legend(handles=hdl, loc='lower center', ncol=3, frameon=False,
            fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.07, 1, 1])

# ─── Figure 2: grid interpolation ───────────────────────────────────────────
print("Fig 2 …")
fig2, axes2 = plt.subplots(1, 3, figsize=(10.5, 3.6))
fig2.patch.set_facecolor('white')

for fi, ax in enumerate(axes2):
    ax_base(ax, col_names[fi], fi)
    u, v = grid_u[fi], grid_v[fi]
    mag  = np.sqrt(u**2 + v**2)
    un, vn = normalize(u, v)

    # grid arrows
    ax.quiver(gx, gy, un, vn, color='#1a2040',
              scale=GRID_RES*0.95, width=0.004,
              headwidth=4, headlength=4.5, headaxislength=3.5,
              alpha=0.88, zorder=3)
    scatter_pts(ax, s=22)

fig2.tight_layout()

# ─── Figure 3: single-feature field with mask ───────────────────────────────
print("Fig 3 …")
fig3, axes3 = plt.subplots(1, 3, figsize=(10.5, 3.6))
fig3.patch.set_facecolor('white')
cw = (xmax-xmin)/GRID_RES
ch = (ymax-ymin)/GRID_RES

for fi, ax in enumerate(axes3):
    ax_base(ax, col_names[fi], fi)
    u, v = grid_u[fi], grid_v[fi]
    mag  = np.sqrt(u**2 + v**2)
    un, vn = normalize(u, v)

    # masked-out cells (light gray)
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            if not unmasked[i, j]:
                rx = xmin + j*cw; ry = ymin + i*ch
                ax.add_patch(plt.Rectangle((rx,ry), cw, ch,
                    fc='#e0e0e0', ec='none', zorder=1))

    # arrows — inside mask only
    mi, mj = np.where(unmasked)
    ax.quiver(gx[mi,mj], gy[mi,mj], un[mi,mj], vn[mi,mj],
              color='#1a1a1a', scale=GRID_RES*0.95, width=0.005,
              headwidth=4, headlength=4.5, headaxislength=3.5,
              alpha=0.9, zorder=4)

    # mask boundary
    ax.contour(gx, gy, unmasked.astype(float), levels=[0.5],
               colors=['#888'], linewidths=1.0, linestyles='--', zorder=5)
    scatter_pts(ax, s=22)

fig3.tight_layout()

# ── Save ────────────────────────────────────────────────────────────────────
os.makedirs('output/paper_figures', exist_ok=True)
for fig, name in [(fig1,'fig1_per_point_gradients'),
                  (fig2,'fig2_grid_interpolation'),
                  (fig3,'fig3_masked_field')]:
    fig.savefig(f'output/paper_figures/{name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(f'output/paper_figures/{name}.png', bbox_inches='tight', dpi=180)
    print(f"  saved {name}")
plt.close('all')
print("Done.")
