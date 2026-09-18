"""3-D renderings in the style of Su et al. (2026).

Two views, matching what the paper shows:

  "cube"      the seismic volume drawn as a solid block with three visible
              faces, fault labels burnt onto them in red — Fig. 16, 19, 22.
  "surfaces"  the fault network itself: every fault's parametric Bezier
              surface, one colour per fault, inside a wireframe box — Fig. 15
              middle panel and Fig. 10(e).

Volumes are stored [z, y, x]; surfaces are stored as (x, y, z) global points.
Depth increases downward, so the z axis is inverted in every plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Qualitative palette for individual fault surfaces (Fig. 15 uses one hue each).
FAULT_COLORS = ["#e8483c", "#f0a63a", "#4ea6d8", "#63bb6b",
                "#a97fd0", "#e07ab0", "#7fd4c8", "#c9b037"]


# --------------------------------------------------------------------------

def load(root: Path, name: str, grid):
    nx, ny, nz = grid
    seis = np.fromfile(root / "seismic" / f"{name}.dat",
                       dtype=np.float32).reshape(nz, ny, nx)
    lab = np.fromfile(root / "labels" / f"{name}.dat",
                      dtype=np.uint8).reshape(nz, ny, nx)
    surfs = None
    sp = root / "surfaces" / f"{name}.npz"
    if sp.exists():
        with np.load(sp) as z:
            surfs = [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
    return seis, lab, surfs


def _face_rgba(seis2d, lab2d, vmin=-2.5, vmax=2.5, label_rgb=(0.93, 0.20, 0.13)):
    g = np.clip((seis2d - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = np.repeat(g[..., None], 3, axis=-1)
    rgba = np.concatenate([rgba, np.ones_like(g)[..., None]], axis=-1)
    if lab2d is not None:
        m = lab2d > 0
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = label_rgb
    return rgba


def best_slices(lab, margin=6, win=9, line_max=18, blob_min=26):
    """Slice indices that cut the faults *cleanly*.

    Outer faces frequently miss the fault network entirely, so the display picks
    interior planes.  Scoring by raw label count is actively harmful: it selects
    planes lying nearly parallel to a fault, where the thin sheet is grazed
    tangentially and its trace spreads into a two-dimensional patch instead of a
    line.

    An earlier version scored crisp pixels by the in-plane distance transform
    and *still* picked exactly those planes: a grazed sheet perturbed by
    `eps_perturb` weaves in and out of the plane, so its footprint is a DOTTED
    scatter, and every isolated dot reads as "thin" to a distance transform.
    The discriminant has to be local *density*, not local thickness — a genuine
    fault trace fills ~9-13 of a 9x9 window, a grazed patch fills most of it.
    """
    from scipy.ndimage import uniform_filter
    nz, ny, nx = lab.shape
    N = win * win

    def score_axis(getter, n):
        best, best_s = n // 2, -np.inf
        for i in range(margin, n - margin):
            m = getter(i)
            if m.sum() < 40:
                continue
            d = uniform_filter(m.astype(np.float32), win, mode="constant") * N
            v = d[m]
            s = float((v <= line_max).sum()) - 3.0 * float((v > blob_min).sum())
            if s > best_s:
                best, best_s = i, s
        return best

    return (score_axis(lambda i: lab[:, :, i] > 0, nx),
            score_axis(lambda i: lab[:, i, :] > 0, ny),
            score_axis(lambda i: lab[i, :, :] > 0, nz))


def _draw_grid(ax, X, Y, Z, facecolors, alpha=None):
    """Vectorised quadrilaterals, preserving one face per seismic pixel."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    P = np.stack([X, Y, Z], axis=-1)
    quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=-2).reshape(-1, 4, 3)
    colors = facecolors[:-1, :-1].reshape(-1, facecolors.shape[-1])
    keep = np.isfinite(quads).all(axis=(1, 2))
    collection = Poly3DCollection(quads[keep], facecolors=colors[keep],
                                  edgecolors="none", linewidths=0,
                                  antialiaseds=False, alpha=alpha)
    ax.add_collection3d(collection)


def draw_cube(ax, seis, lab, grid, show_label=True, at=None):
    """Three orthogonal faces of the volume — the classic seismic block display.

    `at` selects the slice indices (x, y, z).  Pass "outer" for the boundary
    faces of the block, "auto" (the default) to let `best_slices` pick the
    planes that carry the most fault label, or an explicit triple.
    """
    nx, ny, nz = grid
    L = lab if show_label else None
    if at is None or at == "auto":
        ix, iy, iz = best_slices(lab)
    elif at == "outer":
        ix, iy, iz = nx - 1, 0, 0
    else:
        ix, iy, iz = at

    # time slice at z = iz
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    _draw_grid(ax, X, Y, np.full_like(X, iz, dtype=float),
                    facecolors=_face_rgba(seis[iz], None if L is None else L[iz]),
                    alpha=None)

    # inline section at y = iy
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz))
    _draw_grid(ax, X, np.full_like(X, iy, dtype=float), Z,
                    facecolors=_face_rgba(seis[:, iy, :],
                                          None if L is None else L[:, iy, :]),
                    alpha=None)

    # crossline section at x = ix
    Y, Z = np.meshgrid(np.arange(ny), np.arange(nz))
    _draw_grid(ax, np.full_like(Y, ix, dtype=float), Y, Z,
                    facecolors=_face_rgba(seis[:, :, ix],
                                          None if L is None else L[:, :, ix]),
                    alpha=None)
    _wire_box(ax, grid)
    _style_box(ax, grid, **BLOCK_VIEW)


def _lambert(P, base_hex, ambient=0.5, light=(0.45, -0.55, -0.70)):
    """Per-facet colours from a simple Lambert term.

    matplotlib's own `shade=True` leaves visible facet seams on a dense grid,
    so the shading is computed here and passed as `facecolors` instead.
    """
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    l = np.asarray(light, dtype=float)
    l = l / np.linalg.norm(l)
    lam = np.abs(np.einsum("...i,i->...", n, l))
    v = ambient + (1.0 - ambient) * lam
    rgb = np.asarray(matplotlib.colors.to_rgb(base_hex))
    return np.clip(v[..., None] * rgb, 0.0, 1.0)


def draw_surfaces(ax, surfs, grid, alpha=0.8):
    """The fault network as coloured Bezier sheets inside a wireframe box."""
    nx, ny, nz = grid
    for i, pts in enumerate(surfs or []):
        c = FAULT_COLORS[i % len(FAULT_COLORS)]
        inside = ((pts[..., 0] >= -2) & (pts[..., 0] <= nx + 1) &
                  (pts[..., 1] >= -2) & (pts[..., 1] <= ny + 1) &
                  (pts[..., 2] >= -2) & (pts[..., 2] <= nz + 1))
        P = np.where(inside[..., None], pts, np.nan)
        fc = _lambert(np.nan_to_num(P, nan=0.0), c)
        _draw_grid(ax, P[..., 0], P[..., 1], P[..., 2], facecolors=fc, alpha=alpha)
    _wire_box(ax, grid)
    _style_box(ax, grid, **PAPER_VIEW)


def _wire_box(ax, grid):
    nx, ny, nz = grid
    r = [[0, nx - 1], [0, ny - 1], [0, nz - 1]]
    for i in range(2):
        for j in range(2):
            ax.plot([r[0][0], r[0][1]], [r[1][i]] * 2, [r[2][j]] * 2,
                    c="0.55", lw=0.8)
            ax.plot([r[0][i]] * 2, [r[1][0], r[1][1]], [r[2][j]] * 2,
                    c="0.55", lw=0.8)
            ax.plot([r[0][i]] * 2, [r[1][j]] * 2, [r[2][0], r[2][1]],
                    c="0.55", lw=0.8)


#: Viewing angle used for the fault-network panels.  Fig. 15 draws its boxes
#: from a low, wide perspective — broad front face, top face strongly
#: foreshortened, vertical axis on the right — rather than the higher three-
#: quarter view we used first.
PAPER_VIEW = dict(elev=14, azim=-72)
BLOCK_VIEW = dict(elev=22, azim=-58)


def _style_box(ax, grid, elev=22, azim=-58):
    nx, ny, nz = grid
    ax.set_xlim(0, nx - 1); ax.set_ylim(0, ny - 1); ax.set_zlim(nz - 1, 0)
    ax.set_box_aspect((1, 1, 0.85))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("crossline", fontsize=7, labelpad=-4)
    ax.set_ylabel("inline", fontsize=7, labelpad=-4)
    ax.set_zlabel("time", fontsize=7, labelpad=-4, rotation=90)
    ax.tick_params(labelsize=0, length=0, pad=-2)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("0.85")


def draw_section(ax, seis, lab, grid, orient="inline", index=None,
                 vmin=-2.5, vmax=2.5):
    """A single 2-D seismic section with the fault label overlaid in red.

    This is the view a 2-D / 2.5-D model actually consumes, and the one in
    which a fault trace either does or does not sit on a reflector offset.
    """
    nx, ny, nz = grid
    ix, iy, iz = best_slices(lab)
    if orient == "inline":
        i = iy if index is None else index
        s2, l2 = seis[:, i, :], lab[:, i, :]
    elif orient == "crossline":
        i = ix if index is None else index
        s2, l2 = seis[:, :, i], lab[:, :, i]
    else:
        i = iz if index is None else index
        s2, l2 = seis[i, :, :], lab[i, :, :]
    ax.imshow(s2, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    m = np.ma.masked_where(l2 == 0, l2)
    ax.imshow(m, cmap="autumn", aspect="auto", alpha=0.85,
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    return orient, i


# --------------------------------------------------------------------------

def figure_pair(root, name, grid, out_png, title=None):
    """One volume: fault network next to the seismic block it produced."""
    seis, lab, surfs = load(root, name, grid)
    fig = plt.figure(figsize=(11.5, 4.6))
    a1 = fig.add_subplot(131, projection="3d")
    draw_surfaces(a1, surfs, grid)
    a1.set_title("fault network (Bézier surfaces)", fontsize=9)

    a2 = fig.add_subplot(132, projection="3d")
    draw_cube(a2, seis, lab, grid, show_label=True, at="outer")
    a2.set_title("block faces + label (cf. Fig. 16)", fontsize=9)

    ix, iy, iz = best_slices(lab)
    a3 = fig.add_subplot(133, projection="3d")
    draw_cube(a3, seis, lab, grid, show_label=True, at=(ix, iy, iz))
    a3.set_title(f"+ fault label (slices {ix}/{iy}/{iz})", fontsize=9)

    fig.suptitle(title or name, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_png, dpi=155, bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_gallery3(root, names, grid, out_png, orient="inline"):
    """Three columns per category.

      1. fault network, at the paper's viewing angle (Fig. 15)
      2. seismic block, outer faces with the label burnt on (Fig. 16)
      3. the three orthogonal INTERIOR slices, chosen by `best_slices`

    Column 3 is where a fault trace either does or does not sit on a reflector
    offset: the outer faces of column 2 cut the fault network wherever the box
    happens to end, while these planes are picked to cross the faults at a high
    angle.
    """
    n = len(names)
    fig = plt.figure(figsize=(14.0, 3.6 * n))
    for r, name in enumerate(names):
        seis, lab, surfs = load(root, name, grid)
        cat = name.split("_", 1)[1]

        ax = fig.add_subplot(n, 3, r * 3 + 1, projection="3d")
        draw_surfaces(ax, surfs, grid)
        ax.set_title(f"{name} — fault network", fontsize=9)

        ax = fig.add_subplot(n, 3, r * 3 + 2, projection="3d")
        draw_cube(ax, seis, lab, grid, show_label=True, at="outer")
        ax.set_title("seismic block + label", fontsize=9)

        ix, iy, iz = best_slices(lab)
        ax = fig.add_subplot(n, 3, r * 3 + 3, projection="3d")
        draw_cube(ax, seis, lab, grid, show_label=True, at=(ix, iy, iz))
        ax.set_title(f"interior slices {ix}/{iy}/{iz}", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def figure_gallery(root, names, grid, out_png, mode="both"):
    """One row per category: network | seismic+label."""
    ncol = 2 if mode == "both" else 1
    fig = plt.figure(figsize=(5.4 * ncol, 4.3 * len(names)))
    for r, name in enumerate(names):
        seis, lab, surfs = load(root, name, grid)
        cat = name.split("_", 1)[1]
        if mode in ("both", "surfaces"):
            ax = fig.add_subplot(len(names), ncol, r * ncol + 1, projection="3d")
            draw_surfaces(ax, surfs, grid)
            ax.set_title(f"{name} — fault network", fontsize=9)
        if mode in ("both", "cube"):
            ax = fig.add_subplot(len(names), ncol, r * ncol + ncol, projection="3d")
            draw_cube(ax, seis, lab, grid, show_label=True, at="outer")
            ax.set_title(f"{cat} — block faces + label", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/su_repro"))
    p.add_argument("--out", type=Path, default=Path("qc/out/render3d"))
    p.add_argument("--grid", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--random-seed", type=int, default=None,
                   help="pick volumes at random (per category) instead of the first N")
    p.add_argument("--only-3col", action="store_true")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    grid = tuple(a.grid)
    names = sorted(x.stem for x in (a.root / "seismic").glob("*.dat"))
    bycat: dict[str, list[str]] = {}
    for n in names:
        bycat.setdefault(n.split("_", 1)[1], []).append(n)

    if a.random_seed is None:
        picked = [n for cat in sorted(bycat) for n in bycat[cat][:a.per_category]]
    else:
        # A gallery that shows the FIRST volume of each category is not a
        # random sample and should not be read as one; this draws instead.
        rs = np.random.default_rng(a.random_seed)
        picked = [n for cat in sorted(bycat)
                  for n in sorted(rs.choice(bycat[cat], size=min(a.per_category, len(bycat[cat])),
                                            replace=False).tolist())]
        (a.out / "picked.txt").write_text("\n".join(picked) + "\n")
    if not a.only_3col:
        for n in picked:
            print(figure_pair(a.root, n, grid, a.out / f"{n}_3d.png"))
        print(figure_gallery(a.root, picked, grid, a.out / "gallery_3d.png"))
    print(figure_gallery3(a.root, picked, grid, a.out / "gallery_3col.png"))
