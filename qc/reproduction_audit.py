"""Read-only numerical audit of the current Su reproduction.

Writes diagnostics under qc/out/reproduction_audit; never changes the generator,
datasets, or checkpoints. The corrected frame is a geometric reference only.
Run from the project root with the project's existing Python environment.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patterns.su_patterns import CATEGORIES, build_case
from su.geometry import (dip_strike_from_normal, normal_from_points,
                         reference_plane_corners, rotation_matrix)
from su.hull_constraint import _disjoint_mask
from su.displacement import drag_operator
from su.model import GenConfig
from su.optimize import axis_origins, zeta_to_points, objective
from su.topology import tree_from_spec


def plane_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def consistent_frame(n):
    """Invert the actual Eq. 13 frame, preserving the unoriented plane."""
    n = n / np.linalg.norm(n)
    if n[2] > 0:
        n = -n
    dip = np.degrees(np.arctan2(np.hypot(n[0], n[1]), -n[2]))
    strike = np.degrees(np.arctan2(n[1], n[0])) % 360
    return rotation_matrix(strike, dip)


def main():
    out = ROOT / "qc/out/reproduction_audit"
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": "Current source + saved metadata; no production edits"}
    report["roundtrip_examples"] = []
    for strike, dip in [(0, 75), (90, 75), (180, 75), (170, 75), (200, 40)]:
        n = rotation_matrix(strike, dip)[2]
        d, s = dip_strike_from_normal(n)
        report["roundtrip_examples"].append(dict(
            strike=strike, dip=dip, recovered_strike=s, recovered_dip=d,
            error_degrees=plane_angle(n, rotation_matrix(s, d)[2])))

    cube = np.array([[x, y, z] for x in (-10., 10.)
                     for y in (-10., 10.) for z in (-10., 10.)])
    report["hull_containment_counterexample"] = dict(
        triangle_inside_cube=True, expected_disjoint=False,
        actual_disjoint=bool(_disjoint_mask(
            np.array([-1., 0., 0.]), np.array([1., 0., 0.]),
            np.array([1.]), np.array([.5]), cube, 0.)[0, 0]))
    report["drag_near_fault"] = {
        m: drag_operator(np.array([-1e-6, 1e-6]), m).tolist()
        for m in ("normal", "reverse")}
    tree = tree_from_spec([(0, ("P", "P")), (0, ("P", "P"))])
    zeta = np.array([[0., 0., 0.], [1., 3., 1.], [2., 1., 2.]])
    report["missing_P_subtree_counterexample"] = dict(
        tree="root with two (P,P) children", zeta=zeta.tolist(),
        current_objective=objective(tree, zeta, 3),
        sibling_strike_product=float((zeta[1, 0]-zeta[2, 0])*(zeta[1, 1]-zeta[2, 1])),
        explanation="Sibling traces cross in strike despite zero objective; P edges were not removed before applying the sibling case.")

    rows, by_cat = [], {}
    for p in sorted((ROOT / "data/dataset_v1/metadata").glob("*.json")):
        meta = json.loads(p.read_text())
        cat, k = meta["category"], meta["replicate"]
        seed = meta["seed"] + 1000 * CATEGORIES.index(cat) + k
        cfg = GenConfig(grid=tuple(meta["grid"]))
        _, main_fault, _ = build_case(cat, cfg, np.random.default_rng(seed))
        # Saved dimensions are rounded; replay must match before using it.
        f0 = meta["faults"][0]
        assert abs(main_fault.half_len - f0["half_len"]) < .006
        assert abs(main_fault.half_wid - f0["half_wid"]) < .006
        assert abs(main_fault.strike_deg - meta["main"]["strike"]) < 1e-8
        c = np.asarray(main_fault.centre)
        R = rotation_matrix(main_fault.strike_deg, main_fault.dip_deg)
        origins = axis_origins(c, R, reference_plane_corners(
            c, R, main_fault.half_len, main_fault.half_wid))
        dcat = by_cat.setdefault(cat, {"meta": [], "angles": [], "true_dips": [],
                                      "current_dips": [], "residuals": []})
        dcat["meta"].append(meta)
        for fault in meta["faults"][1:]:
            fid = fault["id"]
            pts = zeta_to_points(np.array(meta["zeta"][fid-1]), c, R, origins, cfg.d_z)
            n = normal_from_points(*pts)
            dip, strike = dip_strike_from_normal(n)
            assert abs(dip - fault["dip"]) < .0006
            assert abs(strike - fault["strike"]) < .0006
            current = rotation_matrix(strike, dip)
            fixed = consistent_frame(n)
            err = plane_angle(n, current[2])
            residual = float(np.max(np.abs((pts - pts.mean(0)) @ current[2])))
            true_dip = float(np.degrees(np.arctan2(np.hypot(n[0], n[1]), abs(n[2]))))
            current_dip = min(dip, 180-dip)
            row = dict(sample=p.stem, category=cat, fault=fid,
                       plane_error_degrees=err, point_residual_voxels=residual,
                       true_dip=true_dip, current_dip=current_dip,
                       reference_frame_error_degrees=plane_angle(n, fixed[2]))
            rows.append(row)
            for key, value in [("angles", err), ("true_dips", true_dip),
                               ("current_dips", current_dip), ("residuals", residual)]:
                dcat[key].append(value)

    report["dataset"] = {}
    for cat, values in by_cat.items():
        metas = values["meta"]
        angles = np.array(values["angles"])
        report["dataset"][cat] = dict(
            n_volumes=len(metas), n_branches=len(angles),
            mean_label_percent=100*np.mean([m["fault_fraction"] for m in metas]),
            mean_full_label_percent=100*np.mean([m["fault_fraction_full"] for m in metas]),
            nonzero_pso_count=sum(m["pso"]["penalty"] > 0 for m in metas),
            plane_error_p50=float(np.median(angles)),
            plane_error_p95=float(np.percentile(angles, 95)),
            plane_error_over_10_percent=100*float(np.mean(angles > 10)),
            true_dip_below_20_percent=100*float(np.mean(np.array(values["true_dips"]) < 20)),
            current_dip_below_20_percent=100*float(np.mean(np.array(values["current_dips"]) < 20)),
            point_residual_p50=float(np.median(values["residuals"])))
    report["array_spot_checks"] = []
    for cat in CATEGORIES:
        name = f"000000_{cat}"
        m = json.loads((ROOT/f"data/dataset_v1/metadata/{name}.json").read_text())
        for kind, key in [("labels", "fault_fraction"), ("labels_full", "fault_fraction_full")]:
            a = np.fromfile(ROOT/f"data/dataset_v1/{kind}/{name}.dat", dtype=np.uint8)
            assert a.size == np.prod(m["grid"])
            assert abs(float(a.mean()) - m[key]) < 1e-12
            report["array_spot_checks"].append(dict(sample=name, kind=kind, verified=True))

    report["training_snapshot"] = {}
    for p in sorted((ROOT / "runs").glob("*/log.csv")):
        history = list(csv.DictReader(p.open()))
        if history:
            report["training_snapshot"][p.parent.name] = dict(
                epochs_recorded=len(history),
                best_validation=max(history, key=lambda r: float(r["val_iou"])))
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    with (out / "branch_geometry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist([r["plane_error_degrees"] for r in rows], bins=45, color="#b94b46")
    axes[0].set(xlabel="Angle between optimized and reconstructed planes (degrees)",
                ylabel="Branch count", title=f"Current geometry: {len(rows)} saved branches")
    x = np.arange(len(CATEGORIES))
    ds = report["dataset"]
    axes[1].bar(x-.18, [ds[c]["mean_full_label_percent"] for c in CATEGORIES], .36,
                label="Geometric label")
    axes[1].bar(x+.18, [ds[c]["mean_label_percent"] for c in CATEGORIES], .36,
                label="Filtered training label")
    axes[1].set_xticks(x, ["En echelon", "Horsetail", "Neg. flower", "Pos. flower", "Listric"], rotation=15)
    axes[1].set(ylabel="Mean foreground voxels (%)", title="dataset_v1: identical volumes, different labels")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "audit_summary.png", dpi=160)
    plt.close(fig)

    # A controlled section isolates drag from PSO, geometry and label filtering.
    # It is a mechanism demonstration, not a regenerated dataset example.
    from su.stratigraphy import convolve_z, ricker, sample_reflectivity
    z, x = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    signed = (x - 63.5) / 14
    ref = np.zeros(160, dtype=np.float32)
    ref[np.arange(8, 153, 9)] = np.random.default_rng(7).choice([-1., 1.], 17)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, mode in zip(axes[:2], ["reverse", "normal"]):
        dz = 8 * drag_operator(signed, mode)
        seismic = convolve_z(sample_reflectivity(ref, z + dz), ricker(30, .004))
        ax.imshow(seismic, cmap="gray", vmin=-1, vmax=1, origin="upper")
        ax.axvline(63.5, color="red", linewidth=.7)
        ax.set(xlabel="Horizontal sample", ylabel="Time sample",
               title=f"Current {mode} drag; same labelled plane")
    s = np.linspace(-1, 1, 1001)
    for mode in ("normal", "reverse"):
        # Plot the two sides separately so the jump is not disguised as a ramp.
        for side in (s < 0, s > 0):
            axes[2].plot(s[side], drag_operator(s[side], mode),
                         color={"normal": "#276daa", "reverse": "#ce563f"}[mode],
                         label=mode if side[0] else None)
    axes[2].axvline(0, color="gray", linewidth=.6)
    axes[2].set(xlabel="Signed distance / drag width", ylabel="Drag multiplier",
                title="Fault-plane limits\nexplain the missing jump")
    axes[2].legend()
    fig.suptitle("Single-fault mechanism probe: no noise, no PSO, no label filtering")
    fig.tight_layout()
    fig.savefig(out / "drag_mechanism.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
