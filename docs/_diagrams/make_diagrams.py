"""Generate the diagrams of the documentation (docs/_static/diagrams/).

Run from the repository root::

    python docs/_diagrams/make_diagrams.py            # SVG, light and dark
    python docs/_diagrams/make_diagrams.py --png DIR  # also PNG previews in DIR

Every diagram is drawn twice, ``<name>.svg`` for the light theme and
``<name>_dark.svg`` for the dark one (the ``diagram`` directive of
``docs/_ext/diagrams.py`` shows the one matching the theme). The output is
deterministic: no timestamp, fixed element ids, fixed seeds.

2x2 SPD matrices are drawn as the ellipses {x : x^T X^{-1} x = 1} (the
semi-axes are the square roots of the eigenvalues, along the eigenvectors),
and symmetric (tangent) matrices as signed crosses: one segment per
eigenvector, of length |eigenvalue|, coloured by the sign (blue: positive, red: negative). Everything shown
on an ellipse or a cross is computed exactly (numpy, float64) with the
formulas of the package; the curved surfaces are schematic. The ``assert``
statements check the claims the figures make (unit affine-invariant step,
converged Karcher and Bures-Wasserstein means, centred batch of mean I,
folding example outside the injectivity domain).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


OUT = Path(__file__).resolve().parents[1] / "_static" / "diagrams"
PNG_DIR: Path | None = None

# Palettes: the light / dark text and mark colours of docs/_static/custom.css
# (dataviz categorical slots, contrast >= 4.5:1 on their background).
THEMES = {
    "light": {
        "suffix": "",
        "bg": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "faint": "#d1d9e0",
        "sheet": "#f3efe9",
        "plane": "#2974d0",
        "blue": "#2974d0",
        "orange": "#be542a",
        "green": "#15855d",
        "red": "#c0392b",
        "purple": "#6250d6",
    },
    "dark": {
        "suffix": "_dark",
        "bg": "#1a1a19",
        "ink": "#e8e6e3",
        "muted": "#a7adb5",
        "faint": "#3a4049",
        "sheet": "#262a30",
        "plane": "#6da7ec",
        "blue": "#6da7ec",
        "orange": "#f0995e",
        "green": "#4cc597",
        "red": "#ef7b6b",
        "purple": "#aea6f0",
    },
}
C: dict[str, str] = {}


def _use(theme: str) -> None:
    C.clear()
    C.update(THEMES[theme])
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "text.color": C["ink"],
            "axes.labelcolor": C["ink"],
            "axes.edgecolor": C["muted"],
            "axes.facecolor": C["bg"],
            "figure.facecolor": C["bg"],
            "xtick.color": C["muted"],
            "ytick.color": C["muted"],
            "svg.fonttype": "path",  # text as paths: mathtext renders the same in every browser
            "svg.hashsalt": "yetanotherspdnet",  # stable element ids: no diff noise
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kwargs = {"facecolor": C["bg"], "bbox_inches": "tight", "pad_inches": 0.08}
    fig.savefig(
        OUT / f"{name}{C['suffix']}.svg",
        metadata={"Date": None},  # no timestamp: regenerating is a no-op
        **kwargs,
    )
    if PNG_DIR is not None:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_DIR / f"{name}{C['suffix']}.png", dpi=110, **kwargs)
    plt.close(fig)


# ---------------------------------------------------------------------------
# linear algebra on small symmetric matrices (float64)
# ---------------------------------------------------------------------------
def _eig_fun(m, f):
    w, u = np.linalg.eigh(m)
    return (u * f(w)) @ u.T


def _logm(m):
    return _eig_fun(m, np.log)


def _expm(m):
    return _eig_fun(m, np.exp)


def _powm(m, p):
    return _eig_fun(m, lambda w: w**p)


def _sym(a):
    return (a + a.T) / 2


def _rot(angle_deg):
    a = np.radians(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _spd(eigvals, angle_deg):
    r = _rot(angle_deg)
    return r @ np.diag(eigvals) @ r.T


def _ai_dist(a, b):
    isq = _powm(a, -0.5)
    return np.linalg.norm(_logm(isq @ b @ isq))


def _karcher_mean(batch, tol=1e-12):
    """Affine-invariant (Karcher) mean, iterated to a zero Riemannian gradient."""
    mean = np.mean(batch, axis=0)
    for _ in range(200):
        sq, isq = _powm(mean, 0.5), _powm(mean, -0.5)
        grad = np.mean([_logm(isq @ x @ isq) for x in batch], axis=0)
        mean = sq @ _expm(grad) @ sq
        if np.linalg.norm(grad) < tol:
            break
    assert np.linalg.norm(grad) < 1e-10, "Karcher iteration did not converge"
    return mean


def _bw_cross_sqrt(x, b):
    """(BX)^{1/2} = B^{1/2} (B^{1/2} X B^{1/2})^{1/2} B^{-1/2}."""
    bs, bis = _powm(b, 0.5), _powm(b, -0.5)
    return bs @ _powm(bs @ x @ bs, 0.5) @ bis


def _bw_log(x, b):
    """Log_B(X) = (XB)^{1/2} + (BX)^{1/2} - 2B."""
    t = _bw_cross_sqrt(x, b)
    return _sym(t + t.T - 2 * b)


def _bw_dist2(x, b):
    bs = _powm(b, 0.5)
    return np.trace(x) + np.trace(b) - 2 * np.trace(_powm(bs @ x @ bs, 0.5))


def _bw_mean(batch, tol=1e-13):
    """BW barycenter: fixed point G <- G^{-1/2} (mean (G^{1/2} X G^{1/2})^{1/2})^2 G^{-1/2}."""
    g = np.mean(batch, axis=0)
    for _ in range(500):
        gs, gis = _powm(g, 0.5), _powm(g, -0.5)
        s = np.mean([_powm(gs @ x @ gs, 0.5) for x in batch], axis=0)
        new = gis @ s @ s @ gis
        done = np.linalg.norm(new - g) < tol * np.linalg.norm(g)
        g = new
        if done:
            break
    grad = np.mean([_bw_log(x, g) for x in batch], axis=0)
    assert np.linalg.norm(grad) < 1e-9 * max(1.0, np.linalg.norm(g)), (
        "BW barycenter did not converge"
    )
    return g


def _bw_pt_to_identity(s, b):
    w, u = np.linalg.eigh(b)
    return u @ (np.sqrt(2 / (w[:, None] + w[None, :])) * (u.T @ s @ u)) @ u.T


def _bw_pt_from_identity(s, g):
    w, u = np.linalg.eigh(g)
    return u @ (np.sqrt((w[:, None] + w[None, :]) / 2) * (u.T @ s @ u)) @ u.T


def _bw_exp_identity(s):
    h = np.eye(len(s)) + s / 2
    return h @ h


def _bw_exp(v, g):
    """Exp_G(V) = G + V + Z G Z with G Z + Z G = V."""
    w, u = np.linalg.eigh(g)
    z = u @ ((u.T @ v @ u) / (w[:, None] + w[None, :])) @ u.T
    return _sym(g + v + z @ g @ z)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _arrow(ax, start, end, color=None, style="-|>", lw=1.3, rad=0.0, ls="-", ms=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            color=color or C["muted"],
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def _spd_ellipse(ax, center, matrix, color, alpha=1.0, scale=0.35, lw=1.4, fill=None):
    """Draw the 2x2 SPD matrix as the ellipse {x : x^T M^{-1} x = 1} (scaled)."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * scale * np.sqrt(eigvals[::-1])
    ax.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            fill=fill is not None,
            fc=fill or "none",
            ec=color,
            lw=lw,
            alpha=alpha,
        )
    )


def _sym_cross(ax, center, matrix, scale=0.35, lw=1.6, alpha=1.0):
    """Draw a symmetric 2x2 matrix: one segment per eigenvector, length |eigenvalue|."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    for value, vec in zip(eigvals, eigvecs.T, strict=True):
        d = scale * value * vec
        color = C["blue"] if value >= 0 else C["red"]
        ax.plot(
            [center[0] - d[0], center[0] + d[0]],
            [center[1] - d[1], center[1] + d[1]],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
        )


def _box(ax, x, y, w, h, text, color, fontsize=8.5, fill_alpha=0.10):
    for fc, ec, alpha, lw in ((color, color, fill_alpha, 0), ("none", color, 1.0, 1.3)):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=fc,
                ec=ec,
                alpha=alpha,
                lw=lw,
            )
        )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C["ink"],
        linespacing=1.35,
    )


def _frame(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def daleckii_krein() -> None:
    """Divided differences of ReEig vs the autograd denominators."""
    lam = np.array([1e-3, 1e-3, 1e-3, 0.2, 0.9, 2.5])  # three clamped eigenvalues
    eps = 1e-2
    f = np.maximum(lam, eps)
    fp = (lam > eps).astype(float)
    diff = lam[:, None] - lam[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        loewner = np.where(
            np.abs(diff) < 1e-6,
            fp[:, None] + 0 * diff,
            (f[:, None] - f[None, :]) / diff,
        )
        inv_gap = np.where(np.abs(diff) < 1e-12, np.inf, 1 / np.abs(diff))
    np.fill_diagonal(inv_gap, np.nan)  # the diagonal is not a divided difference
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    im = axes[0].imshow(np.log10(np.minimum(inv_gap, 1e12)), cmap="magma", vmax=12)
    axes[0].set_title(r"autograd: $1/|\lambda_i-\lambda_j|$ ($\log_{10}$)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for (i, j), v in np.ndenumerate(inv_gap):
        if np.isinf(v):
            axes[0].text(
                j, i, "∞", ha="center", va="center", color="black", fontsize=10
            )
    im = axes[1].imshow(loewner, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(r"Daleckii–Krein: $L_{ij}$, $f'(\lambda_i)$ on ties", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(6), [f"{v:g}" for v in lam], rotation=45, fontsize=7)
        ax.set_yticks(range(6), [f"{v:g}" for v in lam], fontsize=7)
    fig.suptitle(
        r"ReEig ($\epsilon = 10^{-2}$) with three equal eigenvalues", fontsize=10
    )
    fig.tight_layout()
    _save(fig, "daleckii_krein")


def implicit_diff() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 2.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 3.1)
    for x, label in (
        (0.3, r"$\Sigma_0$"),
        (1.9, r"$\Sigma_1$"),
        (3.5, "…"),
        (5.1, r"$\Sigma_K$"),
    ):
        ax.text(
            x + 0.5,
            2.2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "fc": "none", "ec": C["blue"]},
        )
    for x in (0.3, 1.9, 3.5):
        _arrow(ax, (x + 0.85, 2.2), (x + 1.75, 2.2), color=C["blue"])
    ax.text(
        3.1,
        2.8,
        r"forward: $\Sigma_{k+1} = F(\Sigma_k, x)$",
        ha="center",
        color=C["blue"],
        fontsize=9,
    )
    ax.text(
        3.1,
        1.25,
        "autograd: backward through the K iterations\n(memory ∝ K)",
        ha="center",
        color=C["muted"],
        fontsize=8.5,
    )
    ax.text(
        8.0,
        2.2,
        r"$\Sigma^\star = F(\Sigma^\star, x)$",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round", "fc": "none", "ec": C["green"]},
    )
    _arrow(ax, (5.95, 2.2), (6.95, 2.2), color=C["green"])
    ax.text(
        8.0,
        1.15,
        "implicit: solve $w = g + J_\\Sigma^\\top w$ once,\n$\\bar x = J_x^\\top w$ (memory independent of K)",
        ha="center",
        color=C["green"],
        fontsize=8.5,
    )
    _save(fig, "implicit_diff")


# ---------------------------------------------------------------------------
# layers and parametrizations
# ---------------------------------------------------------------------------
def parametrization() -> None:
    """Static chart vs moving reference point on a curved manifold."""
    t = np.linspace(-0.2, np.pi + 0.2, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, title in zip(
        axes, ("static: one chart", "dynamic: chart re-centred"), strict=True
    ):
        ax.plot(np.cos(t), np.sin(t), color=C["muted"], lw=1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    p0 = np.array([np.cos(0.35), np.sin(0.35)])
    tangent = np.array([-np.sin(0.35), np.cos(0.35)])
    line = np.array([p0 + s * tangent for s in np.linspace(-0.4, 2.2, 2)])
    axes[0].plot(line[:, 0], line[:, 1], color=C["blue"], lw=1.2, ls="--")
    axes[0].plot(*p0, "o", color=C["blue"])
    axes[0].annotate(
        r"$W_0$", p0, xytext=(8, -12), textcoords="offset points", color=C["blue"]
    )
    far = np.array([np.cos(2.2), np.sin(2.2)])
    proj = p0 + ((far - p0) @ tangent) * tangent
    axes[0].plot(*far, "o", color=C["orange"])
    axes[0].plot([far[0], proj[0]], [far[1], proj[1]], color=C["orange"], lw=1, ls=":")
    axes[0].annotate(
        r"$W$ far from $W_0$:" + "\nchart distorted",
        far,
        xytext=(-64, 16),
        textcoords="offset points",
        color=C["orange"],
        fontsize=8,
    )
    prev = None
    for k, a in enumerate((0.35, 1.05, 1.75, 2.4)):
        p = np.array([np.cos(a), np.sin(a)])
        tg = np.array([-np.sin(a), np.cos(a)])
        seg = np.array([p - 0.35 * tg, p + 0.35 * tg])
        axes[1].plot(seg[:, 0], seg[:, 1], color=C["blue"], lw=1.2, ls="--")
        axes[1].plot(*p, "o", color=C["blue"] if k < 3 else C["orange"])
        if prev is not None:
            _arrow(axes[1], prev * 1.28, p * 1.28, color=C["green"], rad=0.25)
        prev = p
    axes[1].text(
        0.0,
        0.3,
        r"$W_{ref} \leftarrow W$" + "\nevery n steps",
        color=C["green"],
        fontsize=8,
        ha="center",
    )
    _save(fig, "parametrization")


def reeig() -> None:
    """ReEig and ReEigBias as maps of the eigenvalues (two-sided clamp at 1/eps)."""
    eps, bias = 1e-2, 0.03
    lam = np.logspace(-4, 3, 500)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.loglog(lam, lam, color=C["muted"], lw=1, ls=":", label="identity")
    ax.loglog(
        lam,
        np.maximum(lam, eps),
        color=C["blue"],
        lw=2,
        label=r"ReEig $\max(\lambda, \epsilon)$",
    )
    ax.loglog(
        lam,
        np.clip(lam + bias, eps, 1 / eps),
        color=C["orange"],
        lw=1.6,
        ls="--",
        label=r"ReEigBias $\mathrm{clamp}(\lambda + b, \epsilon, 1/\epsilon)$",
    )
    for x, label in ((eps, r"$\epsilon$"), (1 / eps, r"$1/\epsilon$")):
        ax.axvline(x, color=C["muted"], lw=0.8, ls="--")
        ax.text(x * 1.25, 2e-4, label, color=C["muted"])
    ax.set_xlabel(r"eigenvalue $\lambda$ of the input")
    ax.set_ylabel("output eigenvalue")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(r"$\epsilon = 10^{-2}$, $b = 0.03$", fontsize=9)
    _save(fig, "reeig")


def spdnet_pipeline() -> None:
    """SPDNet on HDM05: layers and shapes, manifold part vs Euclidean head."""
    fig, ax = plt.subplots(figsize=(9.8, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3)
    steps = [
        ("X", "(B, 93, 93)", C["muted"]),
        ("BiMap", "(B, 84, 84)", C["blue"]),
        ("ReEig", "(B, 84, 84)", C["blue"]),
        ("BatchNorm", "(B, 84, 84)", C["orange"]),
        ("BiMap\nReEig\nBatchNorm", "(B, 63, 63)", C["blue"]),
        ("LogEig", "(B, 63, 63)", C["green"]),
        ("Vec", "(B, 3969)", C["green"]),
        ("Linear", "(B, 117)", C["green"]),
    ]
    width, gap = 1.25, 0.28
    for i, (name, shape, color) in enumerate(steps):
        x = 0.15 + i * (width + gap)
        _box(
            ax, x, 1.25, width, 0.9, name, color, fontsize=7.5 if "\n" in name else 8.5
        )
        ax.text(
            x + width / 2,
            0.95,
            shape,
            ha="center",
            fontsize=7.5,
            color=C["muted"],
            family="monospace",
        )
        if i:
            _arrow(ax, (x - gap + 0.02, 1.7), (x - 0.02, 1.7))
    x_log = 0.15 + 5 * (width + gap)
    ax.plot([0.15, x_log - 0.14], [2.5, 2.5], color=C["blue"], lw=1.2)
    ax.text(
        (0.15 + x_log) / 2,
        2.62,
        "on the SPD manifold: every output is SPD",
        ha="center",
        color=C["blue"],
        fontsize=8.5,
    )
    ax.plot([x_log, 12.05], [2.5, 2.5], color=C["green"], lw=1.2)
    ax.text(
        (x_log + 12.05) / 2,
        2.62,
        "flat space: Euclidean head",
        ha="center",
        color=C["green"],
        fontsize=8.5,
    )
    ax.text(
        6.1,
        0.25,
        "SPDnet(input_dim=93, hidden_layers_size=[84, 63], output_dim=117, batchnorm=True)",
        ha="center",
        fontsize=8,
        color=C["muted"],
        family="monospace",
    )
    _save(fig, "spdnet_pipeline")


def package_levels() -> None:
    """The levels of the package (and random) and the neighbouring repositories."""
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    levels = [
        (
            3.0,
            "model",
            "SPDnet · RResNet · GBWBNRResNet",
            "classifiers, one constructor",
            C["orange"],
        ),
        (
            1.7,
            "nn",
            "BiMap · ReEig · LogEig · BatchNorm · ResidualBlock ·\nSampleCovariance · MEstimation · parametrizations",
            "modules with parameters",
            C["blue"],
        ),
        (
            0.3,
            "functions",
            "spd_linalg · spd_geometries/* · m_estimators ·\nstiefel · scalar_functions · dispatch",
            "pure tensor functions, two gradient paths",
            C["green"],
        ),
    ]
    for y, name, content, role, color in levels:
        _box(ax, 2.0, y, 5.4, 1.0, "", color)
        ax.text(2.18, y + 0.74, name, fontsize=10, color=color, weight="bold")
        ax.text(
            4.7,
            y + 0.4,
            content,
            ha="center",
            va="center",
            fontsize=7.6,
            color=C["ink"],
            family="monospace",
        )
        ax.text(
            7.25, y + 0.74, role, ha="right", fontsize=7.5, color=color, style="italic"
        )
    for y0, y1 in ((3.0, 2.7), (1.7, 1.3)):
        _arrow(ax, (4.7, y0 - 0.02), (4.7, y1 + 0.02))
    # random: generators for the tests, the examples and synthetic data
    _box(ax, 7.75, 0.3, 2.2, 1.0, "", C["purple"])
    ax.text(7.9, 1.04, "random", fontsize=10, color=C["purple"], weight="bold")
    ax.text(
        8.85,
        0.58,
        "random_SPD ·\nrandom_stiefel",
        ha="center",
        va="center",
        fontsize=7.4,
        family="monospace",
        color=C["ink"],
    )
    _box(
        ax,
        0.05,
        1.7,
        1.6,
        1.0,
        "spdnet-datasets\n(loaders, SPD data)",
        C["muted"],
        fontsize=7.4,
        fill_alpha=0.05,
    )
    _arrow(ax, (1.7, 2.2), (1.95, 2.2))
    _box(
        ax,
        7.75,
        2.35,
        2.2,
        1.0,
        "spdnet-training\n(Lightning, Hydra,\nbackbones)",
        C["muted"],
        fontsize=7.4,
        fill_alpha=0.05,
    )
    _arrow(ax, (7.7, 2.85), (7.45, 2.85))
    ax.text(
        5.0,
        4.2,
        "yetanotherspdnet",
        ha="center",
        fontsize=11,
        color=C["ink"],
        weight="bold",
    )
    _save(fig, "package_levels")


# ---------------------------------------------------------------------------
# batch normalization
# ---------------------------------------------------------------------------
def _toy_batch(n=7, seed=3, base=((2.2, 0.9), (0.9, 0.8)), spread=0.35):
    rng = np.random.default_rng(seed)
    base = np.array(base)
    out = []
    for _ in range(n):
        a = rng.normal(size=(2, 2)) * spread
        out.append(base + a @ a.T)
    return np.array(out)


def batchnorm_steps() -> None:
    """Affine-invariant batch norm on 2x2 matrices: Karcher mean, centring, scaling, bias."""
    batch = _toy_batch()
    mean = _karcher_mean(batch)
    isq = _powm(mean, -0.5)
    centred = np.array([isq @ m @ isq for m in batch])
    assert np.linalg.norm(_karcher_mean(centred) - np.eye(2)) < 1e-9
    sigma = np.sqrt(np.mean([_ai_dist(np.eye(2), c) ** 2 for c in centred]))
    s = 0.6
    scaled = np.array([_powm(c, s / sigma) for c in centred])
    g = np.array([[0.85, -0.25], [-0.25, 1.2]])  # the bias parameter G
    biased = np.array(
        [g @ c @ g for c in scaled]
    )  # whitening mode: the mean goes to G^2
    stages = [
        (batch, r"batch $X_i$", mean, r"Karcher mean $\bar X$"),
        (centred, "centred\n" + r"$\bar X^{-1/2} X_i \bar X^{-1/2}$", np.eye(2), "$I$"),
        (
            scaled,
            "rescaled\n" + rf"$(\cdot)^{{s/\sigma}}$, $\sigma = {sigma:.2f}$",
            np.eye(2),
            "$I$",
        ),
        (biased, "biased\n" + r"$G\,(\cdot)\,G$", g @ g, "$G^2$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.9))
    for ax, (mats, title, ref, ref_label) in zip(axes, stages, strict=True):
        for m in mats:
            _spd_ellipse(ax, (0, 0), m, C["blue"], alpha=0.65, lw=1.1)
        _spd_ellipse(ax, (0, 0), ref, C["orange"], lw=2.4)
        ax.text(0, -1.0, ref_label, color=C["orange"], ha="center", fontsize=11)
        _frame(ax, (-1.1, 1.1), (-1.15, 1.1))
        ax.set_title(title, fontsize=11)
    for i in range(3):
        fig.text(
            0.262 + 0.243 * i, 0.47, "→", fontsize=18, color=C["muted"], ha="center"
        )
    fig.tight_layout()
    _save(fig, "batchnorm_steps")


GBWBN_BASE = ((1.6, 0.5), (0.5, 0.7))


def _gbwbn_example(scale=1.0, theta=0.5):
    """GBWBN on a 2x2 batch, step by step (the formulas of the package)."""
    batch = scale * _toy_batch(n=6, seed=5, base=GBWBN_BASE, spread=0.75)
    m = _spd((1.3, 0.8), 25)  # learned metric M
    g = _spd((1.4, 0.7), -30)  # learned bias G (input space)
    m_isq, m_sq = _powm(m, -0.5), _powm(m, 0.5)
    hat = np.array([m_isq @ _powm(x, theta) @ m_isq for x in batch])
    b = _bw_mean(hat)
    logs = np.array([_bw_log(x, b) for x in hat])
    at_identity = np.array([_bw_pt_to_identity(v, b) for v in logs])
    sigma = np.sqrt(np.mean([_bw_dist2(x, b) for x in hat])) / abs(theta)
    s = 0.6
    factor = s / np.sqrt(sigma**2 + 1e-5)
    scaled = factor * at_identity
    g_hat = m_isq @ _powm(g, theta) @ m_isq
    at_g = np.array([_bw_pt_from_identity(v, g_hat) for v in scaled])
    biased = np.array([_bw_exp(v, g_hat) for v in at_g])
    out = np.array([_powm(m_sq @ y @ m_sq, 1 / theta) for y in biased])
    return {
        "batch": batch,
        "M": m,
        "G": g,
        "hat": hat,
        "B": b,
        "logs": logs,
        "at_identity": at_identity,
        "scaled": scaled,
        "factor": factor,
        "g_hat": g_hat,
        "at_g": at_g,
        "biased": biased,
        "out": out,
    }


def gbwbn_steps() -> None:
    """GBWBN (Wang et al. 2025, Algorithm 1) on 2x2 matrices, step by step."""
    e = _gbwbn_example()
    # centred batch: no folding here, and its BW barycenter is exactly I
    centred = np.array([_bw_exp_identity(v) for v in e["at_identity"]])
    assert (
        min(np.linalg.eigvalsh(np.eye(2) + v / 2).min() for v in e["at_identity"]) > 0
    )
    assert np.linalg.norm(_bw_mean(centred) - np.eye(2)) < 1e-8
    sc = 0.33
    panels = [
        ("points", e["batch"], e["M"], r"$M$", "1. input $X_i$\nand metric $M$"),
        (
            "points",
            e["hat"],
            e["B"],
            r"$B$",
            "2. deformation\n"
            + r"$\hat X_i = M^{-1/2} X_i^{\theta} M^{-1/2}$"
            + "\nand their barycenter $B$",
        ),
        (
            "vectors",
            e["logs"],
            e["B"],
            r"$B$",
            "3. tangent vectors at $B$\n" + r"$\mathrm{Log}_B(\hat X_i)$",
        ),
        (
            "vectors",
            e["at_identity"],
            np.eye(2),
            r"$I$",
            "4. parallel transport\n" + r"$\Gamma_{B\to I}$ to the identity",
        ),
        (
            "vectors",
            e["scaled"],
            np.eye(2),
            r"$I$",
            "5. scaling at $I$\n" + rf"$\times\, s/\sigma = {e['factor']:.2f}$",
        ),
        (
            "vectors",
            e["at_g"],
            e["g_hat"],
            r"$\hat G$",
            "6. parallel transport "
            + r"$\Gamma_{I\to\hat G}$"
            + "\n"
            + r"$\hat G = M^{-1/2} G^{\theta} M^{-1/2}$",
        ),
        (
            "points",
            e["biased"],
            e["g_hat"],
            r"$\hat G$",
            "7. exponential\n" + r"$\mathrm{Exp}_{\hat G}(\cdot)$",
        ),
        (
            "points",
            e["out"],
            e["G"],
            r"$G$",
            "8. inverse deformation\n"
            + r"$Y_i = (M^{1/2} (\cdot) M^{1/2})^{1/\theta}$",
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 6.6))
    for ax, (kind, mats, ref, ref_label, title) in zip(axes.flat, panels, strict=True):
        if kind == "points":
            for x in mats:
                _spd_ellipse(ax, (0, 0), x, C["blue"], alpha=0.7, lw=1.1, scale=sc)
            _spd_ellipse(ax, (0, 0), ref, C["orange"], lw=2.4, scale=sc)
        else:
            _spd_ellipse(ax, (0, 0), ref, C["orange"], lw=2.0, scale=sc, alpha=0.5)
            for v in mats:
                _sym_cross(ax, (0, 0), v, scale=0.75, lw=1.5, alpha=0.85)
        ax.text(0, -0.98, ref_label, color=C["orange"], ha="center", fontsize=12)
        _frame(ax, (-1.05, 1.05), (-1.1, 1.0))
        ax.set_title(title, fontsize=11.5, linespacing=1.3)
    for row in range(2):
        for i in range(3):
            fig.text(
                0.265 + 0.24 * i,
                0.70 - 0.46 * row,
                "→",
                fontsize=17,
                color=C["muted"],
                ha="center",
            )
    fig.tight_layout(h_pad=2.0)
    _save(fig, "gbwbn_steps")


def _min_eig_centring(batch, theta, trace_scale):
    hat = np.array([_powm(x, theta) for x in batch])
    if trace_scale:  # M = cI with c = tr(B) / n: data of unit scale
        hat = hat / (np.trace(_bw_mean(hat)) / len(hat[0]))
    b = _bw_mean(hat)
    return min(
        np.linalg.eigvalsh(np.eye(2) + _bw_pt_to_identity(_bw_log(x, b), b) / 2).min()
        for x in hat
    )


def gbwbn_folding() -> None:
    """Folding of Exp_I(V) = (I + V/2)^2: the scalar picture, a 2x2 example, scale and spread."""
    fig, axes = plt.subplots(
        1, 3, figsize=(10.8, 3.4), gridspec_kw={"width_ratios": [1.05, 0.85, 1.2]}
    )
    # (a) scalar picture
    ax = axes[0]
    v = np.linspace(-5, 2.5, 400)
    ax.plot(v, (1 + v / 2) ** 2, color=C["blue"], lw=2)
    ax.axvspan(-5, -2, color=C["red"], alpha=0.12, lw=0)
    ax.axvline(-2, color=C["red"], lw=1, ls="--")
    ax.text(-4.85, 5.1, "folded:\n$1 + v/2 < 0$", color=C["red"], fontsize=8)
    ax.text(-1.7, 5.1, "injective:\n$1 + v/2 > 0$", color=C["blue"], fontsize=8)
    ax.plot([-3, 1], [0.25, 2.25], "o", color=C["orange"], ms=4)
    ax.annotate(
        "",
        (1, 2.25),
        (-3, 0.25),
        arrowprops={"arrowstyle": "<->", "color": C["orange"], "lw": 0.8, "ls": ":"},
    )
    ax.text(-1.4, 2.55, "same point", color=C["orange"], fontsize=7.5)
    ax.plot(-2, 0, "o", color=C["red"], ms=5)
    ax.set_xlabel(r"eigenvalue $v$ of $V = \Gamma_{B\to I}\,\mathrm{Log}_B(\hat X)$")
    ax.set_ylabel(r"$(1 + v/2)^2$")
    ax.set_ylim(-0.3, 6.4)
    ax.set_title(r"(a) $\mathrm{Exp}_I$ on one eigenvalue", fontsize=9)
    # (b) a folded 2x2 matrix: data at scale 100, theta = 1
    ax = axes[1]
    e = _gbwbn_example(scale=100.0, theta=1.0)
    k = int(
        np.argmin(
            [np.linalg.eigvalsh(np.eye(2) + u / 2).min() for u in e["at_identity"]]
        )
    )
    vec = e["at_identity"][k]
    lam_min = np.linalg.eigvalsh(np.eye(2) + vec / 2).min()
    assert lam_min <= 0, "the folding example must leave the injectivity domain"
    folded = _bw_exp_identity(vec)
    w, u = np.linalg.eigh(np.eye(2) + vec / 2)
    other = 2 * (
        u @ np.diag(np.abs(w)) @ u.T - np.eye(2)
    )  # the preimage inside the domain
    assert np.allclose(_bw_exp_identity(other), folded)
    sc = 0.2
    _spd_ellipse(ax, (0, -0.25), np.eye(2), C["orange"], lw=2.0, scale=sc)
    _spd_ellipse(ax, (0, -0.25), folded, C["red"], lw=1.8, scale=sc)
    ax.text(
        0.0,
        -1.55,
        r"$\mathrm{Exp}_I(V) = \mathrm{Exp}_I(V')$",
        ha="center",
        color=C["red"],
        fontsize=8.5,
    )
    _sym_cross(ax, (-1.05, 1.1), vec, scale=0.06, lw=1.6)
    ax.text(-1.05, 0.55, "$V$ (outside)", ha="center", fontsize=8)
    _sym_cross(ax, (1.05, 1.1), other, scale=0.06, lw=1.6)
    ax.text(1.05, 0.55, "$V'$ (inside)", ha="center", fontsize=8)
    ax.text(
        0,
        1.75,
        rf"$\lambda_{{\min}}(I + V/2) = {lam_min:.2f}$",
        ha="center",
        fontsize=8.5,
        color=C["red"],
    )
    _frame(ax, (-1.7, 1.7), (-1.75, 1.95))
    ax.set_title(r"(b) data $\times 100$, $\theta = 1$: folded", fontsize=9)
    # (c) the smallest eigenvalue of I + V/2 against the scale of the data
    ax = axes[2]
    scales = np.logspace(-2, 4, 49)
    narrow = _toy_batch(n=6, seed=5, base=GBWBN_BASE, spread=0.75)
    wide = np.array([_spd((0.01 * (1 + k), 100.0 / (1 + k)), 30 * k) for k in range(6)])
    curves = [
        (narrow, 1.0, False, r"$\theta = 1$", C["blue"], "-"),
        (narrow, 0.5, False, r"$\theta = 1/2$", C["green"], "-"),
        (narrow, 1.0, True, r"$\theta = 1$, trace_scale", C["purple"], "--"),
        (
            wide,
            1.0,
            True,
            r"spread $\kappa \approx 10^{4}$, $\theta = 1$, trace_scale",
            C["orange"],
            "--",
        ),
        (
            wide,
            0.5,
            True,
            r"spread $\kappa \approx 10^{4}$, $\theta = 1/2$, trace_scale",
            C["orange"],
            ":",
        ),
    ]
    for batch, theta, trace_scale, label, color, ls in curves:
        values = [_min_eig_centring(c * batch, theta, trace_scale) for c in scales]
        ax.semilogx(scales, values, color=color, lw=1.8, ls=ls, label=label)
    ax.axhspan(-6, 0, color=C["red"], alpha=0.10, lw=0)
    ax.axhline(0, color=C["red"], lw=0.9, ls="--")
    ax.text(1.4e-2, -0.35, "folding", color=C["red"], fontsize=8)
    ax.set_ylim(-2.5, 1.15)
    ax.set_xlabel(r"scale $c$ of the data ($c\,X_i$)")
    ax.set_ylabel(r"$\min_i \lambda_{\min}(I + V_i/2)$")
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    ax.set_title("(c) centring step: scale and spread", fontsize=9)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "gbwbn_folding")


# ---------------------------------------------------------------------------
# residual networks (RResNet)
# ---------------------------------------------------------------------------
# The example block: X_i (2x2), the field f(lambda) and the learned frame Q.
X_I = _spd((0.45, 2.2), 30)
Q_ANGLE = 75.0
F_LAMBDA = np.array(
    [0.9, -0.55]
)  # f_theta at the two positions of the ascending spectrum


def _rresnet_step(x=X_I, f=F_LAMBDA, q_angle=Q_ANGLE):
    q = _rot(q_angle)
    v = q @ np.diag(f) @ q.T
    isq, sq = _powm(x, -0.5), _powm(x, 0.5)
    w = isq @ v @ isq
    nu = np.linalg.norm(w)
    v_hat = v / nu
    w_hat = w / nu
    x_next = sq @ _expm(w_hat) @ sq
    assert abs(_ai_dist(x, x_next) - 1.0) < 1e-12, "the AI step must have unit length"
    path = [sq @ _expm(t * w_hat) @ sq for t in (0, 0.25, 0.5, 0.75, 1.0)]
    return {
        "V": v,
        "V_hat": v_hat,
        "W_hat": w_hat,
        "nu": nu,
        "next": x_next,
        "path": path,
    }


def rresnet_pipeline() -> None:
    """The residual block as a pipeline: one eigh, field, normalisation, exponential."""
    step = _rresnet_step()
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.axis("off")
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 5.2)
    bw, bh = 2.75, 1.3
    _spd_ellipse(ax, (0.6, 3.65), X_I, C["ink"], lw=1.8, scale=0.4)
    ax.text(0.6, 2.82, "$X_i$", ha="center", fontsize=11)
    row1 = [
        (
            1.35,
            "(1) floor + eigh\n"
            + r"$X_i = U \Lambda U^\top$"
            + "\n"
            + r"$\tilde\lambda = \max(\lambda, \epsilon)$",
            C["blue"],
        ),
        (
            4.55,
            "spectral field $f_\\theta$\nConv1d (+ BatchNorm1d)\nalong the ascending $\\tilde\\lambda$",
            C["purple"],
        ),
        (
            7.75,
            "tangent vector\n"
            + r"$V = Q\,\mathrm{diag}(f(\tilde\lambda))\,Q^\top$"
            + "\n$Q$ learned (Stiefel)",
            C["purple"],
        ),
    ]
    for x, text, color in row1:
        _box(ax, x, 3.0, bw, bh, text, color, fontsize=8.3)
    _arrow(ax, (1.05, 3.65), (1.33, 3.65))
    for x0, label in ((4.1, r"$\tilde\lambda$"), (7.3, r"$f(\tilde\lambda)$")):
        _arrow(ax, (x0 + 0.02, 3.65), (x0 + 0.43, 3.65))
        ax.text(x0 + 0.22, 3.85, label, ha="center", fontsize=9, color=C["muted"])
    _sym_cross(ax, (11.35, 3.65), step["V"], scale=0.45, lw=2.0)
    ax.text(11.35, 2.95, "$V$", ha="center", fontsize=10)
    _arrow(ax, (10.52, 3.65), (10.85, 3.65))
    row2 = [
        (
            7.75,
            "unit step\n"
            + r"$\hat V = V / \|V\|_{X_i}$"
            + "\n"
            + r"$\|V\|_{X_i} = \|X_i^{-1/2} V X_i^{-1/2}\|_F$",
            C["green"],
        ),
        (
            10.95,
            "(2) exponential\n"
            + r"$X_i^{1/2}\exp(X_i^{-1/2}\hat V X_i^{-1/2})\,X_i^{1/2}$"
            + "\n"
            + r"$= \mathrm{Exp}_{X_i}(\hat V)$",
            C["green"],
        ),
    ]
    for x, text, color in row2:
        _box(ax, x, 0.95, bw + 0.2 if x > 10 else bw, bh, text, color, fontsize=8.3)
    _arrow(ax, (11.35, 2.8), (10.0, 2.3), rad=-0.25)
    _arrow(ax, (10.52, 1.6), (10.93, 1.6))
    _spd_ellipse(ax, (14.05, 3.65), step["next"], C["ink"], lw=1.8, scale=0.4)
    ax.text(14.05, 2.82, "$X_{i+1}$", ha="center", fontsize=11)
    _arrow(ax, (13.95, 2.3), (14.05, 2.62))
    # skip connection: X_i is the base point of the step
    ax.plot([0.6, 0.6], [2.6, 0.42], color=C["orange"], lw=1.6)
    ax.plot([0.6, 12.4], [0.42, 0.42], color=C["orange"], lw=1.6)
    _arrow(ax, (12.4, 0.42), (12.4, 0.93), color=C["orange"], lw=1.6)
    ax.text(
        0.8,
        0.6,
        r"skip connection: $X_i$ is the base point of the step;"
        + "\n"
        + r"$X_i^{\pm 1/2} = U \tilde\Lambda^{\pm 1/2} U^\top$ reuse eigh (1)",
        ha="left",
        fontsize=8.3,
        color=C["orange"],
    )
    ax.text(
        7.3,
        4.85,
        "two eigendecompositions per block, (1) and (2), each with a Daleckii–Krein backward",
        ha="center",
        fontsize=8.4,
        color=C["muted"],
    )
    _save(fig, "rresnet_pipeline")


def rresnet_spectrum() -> None:
    """Eigenvalue view: floored spectrum, signed field, learned frame Q vs the eigenvectors of X."""
    lam = np.array(
        [3e-9, 4e-4, 6e-3, 0.03, 0.2, 0.9, 2.5, 7.0]
    )  # ascending, one below the floor
    floor = max(1e-8, np.finfo(np.float64).eps * lam.max())
    lam_t = np.maximum(lam, floor)
    # an example field: Conv1d of kernel 3 along the sorted raw spectrum, example weights
    kernel, bias = np.array([0.55, -0.8, 0.35]), 0.12
    padded = np.concatenate([[0.0], lam_t, [0.0]])
    f = np.tanh(
        np.array([kernel @ padded[k : k + 3] for k in range(len(lam_t))]) + bias
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(10.8, 3.3), gridspec_kw={"width_ratios": [1.15, 1.15, 0.95]}
    )
    ax = axes[0]
    idx = np.arange(1, len(lam) + 1)
    ax.bar(idx, lam_t, color=C["blue"], alpha=0.85, width=0.65, log=True)
    ax.bar(
        idx[:1],
        lam_t[:1],
        color="none",
        edgecolor=C["orange"],
        hatch="///",
        width=0.65,
        log=True,
        lw=1.2,
    )
    ax.plot(idx[:1], lam[:1], "v", color=C["orange"], ms=6)
    ax.axhline(floor, color=C["orange"], lw=1.0, ls="--")
    ax.text(
        0.45,
        5e2,
        "dashed: floor $\\epsilon = \\max(10^{-8}, \\varepsilon\\,\\lambda_{\\max})$\n"
        "hatched: $\\lambda_1 = 3\\cdot 10^{-9}$ raised to $\\epsilon$",
        fontsize=7.4,
        color=C["orange"],
        va="top",
    )
    ax.set_ylim(1e-9, 2e3)
    ax.set_xticks(idx)
    ax.set_xlabel(r"position $k$ in the ascending spectrum")
    ax.set_ylabel(r"$\tilde\lambda_k$ (log scale)")
    ax.set_title(r"(a) floored spectrum $\tilde\lambda$ (one eigh)", fontsize=9)
    ax = axes[1]
    colors = [C["blue"] if val >= 0 else C["red"] for val in f]
    ax.bar(idx, f, color=colors, width=0.65)
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(idx)
    ax.set_xlabel(r"position $k$")
    ax.set_ylabel(r"$f_\theta(\tilde\lambda)_k$")
    ax.set_title(
        "(b) field: Conv1d along the positions\n(kernel 3, example weights, tanh)",
        fontsize=9,
    )
    ax = axes[2]
    _spd_ellipse(
        ax, (0, 0), X_I, C["muted"], lw=1.4, scale=0.5, fill=C["faint"], alpha=0.9
    )
    w, u = np.linalg.eigh(X_I)
    for k in range(2):
        d = 0.5 * np.sqrt(w[k]) * u[:, k]
        ax.plot([-d[0], d[0]], [-d[1], d[1]], color=C["ink"], lw=1.2, ls=":")
    q = _rot(Q_ANGLE)
    for k in range(2):
        d = 0.85 * abs(F_LAMBDA[k]) * q[:, k]
        color = C["blue"] if F_LAMBDA[k] >= 0 else C["red"]
        _arrow(ax, (0, 0), (d[0], d[1]), color=color, lw=1.8)
        _arrow(ax, (0, 0), (-d[0], -d[1]), color=color, lw=1.8)
    ax.text(
        -1.05,
        1.08,
        "dotted: eigenvectors $U$ of $X_i$\narrows: $|f_k|\\,q_k$ ($Q$)",
        fontsize=7.6,
        va="top",
        color=C["muted"],
    )
    ax.text(
        0,
        -1.05,
        r"$V = Q\,\mathrm{diag}(f)\,Q^\top$, $Q \neq U$",
        ha="center",
        fontsize=8.5,
    )
    _frame(ax, (-1.05, 1.05), (-1.15, 1.1))
    ax.set_title("(c) learned frame $Q$ vs $U$ (2 × 2)", fontsize=9)
    fig.tight_layout(w_pad=1.6)
    _save(fig, "rresnet_spectrum")


def _surface(x, y):
    return 0.95 * np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2) / 7.0)


def _project(x, y, z):
    """Oblique projection of the schematic surface."""
    return x + 0.55 * y, 0.42 * y + 1.6 * z


def rresnet_manifold() -> None:
    """The AI step on the manifold: tangent plane, unit ball, geodesic, whitened inset."""
    step = _rresnet_step()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    _frame(ax, (-4.0, 5.6), (-1.95, 3.7))
    grid = np.linspace(-3, 3, 13)
    fine = np.linspace(-3, 3, 120)
    outline = (
        [(-3, y) for y in fine]
        + [(x, 3) for x in fine]
        + [(3, y) for y in fine[::-1]]
        + [(x, -3) for x in fine[::-1]]
    )
    ax.add_patch(
        Polygon(
            [_project(x, y, _surface(x, y)) for x, y in outline],
            closed=True,
            fc=C["sheet"],
            ec="none",
        )
    )
    for g in grid:
        ax.plot(
            *_project(np.full_like(fine, g), fine, _surface(g, fine)),
            color=C["faint"],
            lw=0.8,
        )
        ax.plot(
            *_project(fine, np.full_like(fine, g), _surface(fine, g)),
            color=C["faint"],
            lw=0.8,
        )
    ax.text(
        *_project(3.0, -3.0, _surface(3.0, -3.0)),
        r"$\mathcal{S}^{++}_n$  ",
        fontsize=12,
        color=C["muted"],
        ha="right",
        va="top",
    )
    # base point and tangent plane
    x0, y0 = -0.9, -0.9
    z0 = _surface(x0, y0)
    h = 1e-6
    zx = (_surface(x0 + h, y0) - _surface(x0 - h, y0)) / (2 * h)
    zy = (_surface(x0, y0 + h) - _surface(x0, y0 - h)) / (2 * h)

    def plane(a, b, size=1.0):
        px, py = x0 + size * a, y0 + size * b
        return _project(px, py, z0 + zx * (px - x0) + zy * (py - y0))

    corners = [plane(a, b, 1.9) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(
        Polygon(corners, closed=True, fc=C["plane"], alpha=0.10, ec=C["plane"], lw=1.0)
    )
    ax.text(
        *plane(-1.0, 1.0, 1.9),
        r" $T_{X_i}\mathcal{S}^{++}_n$",
        fontsize=10.5,
        color=C["plane"],
        va="bottom",
    )
    # exact slice of the tangent space: Frobenius-orthonormal E1 (along V_hat), E2
    e1 = step["V_hat"] / np.linalg.norm(step["V_hat"])
    other = np.array([[0.0, 1.0], [1.0, 0.0]])
    e2 = other - np.sum(other * e1) * e1
    e2 /= np.linalg.norm(e2)
    x_inv = np.linalg.inv(X_I)
    basis = (e1, e2)
    gram = np.array([[np.trace(x_inv @ p @ x_inv @ q) for q in basis] for p in basis])
    t = np.linspace(0, 2 * np.pi, 200)
    gw, gu = np.linalg.eigh(gram)
    ell = gu @ np.diag(1 / np.sqrt(gw)) @ np.vstack([np.cos(t), np.sin(t)])
    unit = 1.25  # plane units per Frobenius unit
    ax.plot(
        *np.array([plane(a, b, unit) for a, b in ell.T]).T, color=C["green"], lw=1.8
    )
    ax.plot(
        *np.array([plane(np.cos(s), np.sin(s), unit) for s in t]).T,
        color=C["muted"],
        lw=0.9,
        ls="--",
    )
    a_hat = np.linalg.norm(step["V_hat"])  # V_hat = a_hat * E1
    assert (
        abs(np.sqrt(gram[0, 0]) * a_hat - 1.0) < 1e-12
    )  # V_hat is on the AI unit sphere
    p0 = plane(0, 0)
    tip = plane(a_hat, 0, unit)
    _arrow(ax, p0, tip, color=C["green"], lw=2.2, ms=14)
    ax.text(
        tip[0] - 0.05,
        tip[1] + 0.32,
        r"$\hat V$",
        color=C["green"],
        fontsize=11,
        ha="right",
    )
    lab = plane(0.2, -1.55, unit)
    ax.text(
        lab[0],
        lab[1],
        r"AI unit ball $\{\|X_i^{-1/2} V X_i^{-1/2}\|_F = 1\}$",
        color=C["green"],
        fontsize=8.5,
        ha="center",
    )
    lab = plane(-1.35, 0.85, unit)
    ax.text(
        lab[0],
        lab[1],
        "Frobenius\nunit circle",
        color=C["muted"],
        fontsize=7.5,
        ha="center",
    )
    # geodesic on the surface, leaving X_i along V_hat, with the exact ellipses
    span = 2.7
    direction = np.array([1.0, 0.0])
    ts = np.linspace(0, 1, 60)
    gx, gy = x0 + span * ts * direction[0], y0 + span * ts * direction[1] + 0.35 * ts**2
    ax.plot(*_project(gx, gy, _surface(gx, gy)), color=C["orange"], lw=2.4)
    for k, tk in enumerate((0, 0.25, 0.5, 0.75, 1.0)):
        px, py = x0 + span * tk, y0 + 0.35 * tk**2
        sx, sy = _project(px, py, _surface(px, py))
        ax.plot(sx, sy, "o", color=C["orange"], ms=4.5)
        ax.plot(
            [sx, sx + 0.05], [sy + 0.08, sy + 0.62], color=C["muted"], lw=0.7, ls=":"
        )
        _spd_ellipse(
            ax,
            (sx + 0.05, sy + 0.95),
            step["path"][k],
            C["ink"],
            lw=1.4,
            scale=0.2,
            fill=C["bg"],
        )
        ax.text(
            sx + 0.05,
            sy + 1.38,
            f"t = {tk:g}",
            ha="center",
            fontsize=7.5,
            color=C["muted"],
        )
    sx, sy = _project(x0, y0, z0)
    ax.text(sx - 0.12, sy - 0.1, "$X_i$", ha="right", va="top", fontsize=11)
    end = _project(x0 + span, y0 + 0.35, _surface(x0 + span, y0 + 0.35))
    ax.text(end[0] + 0.15, end[1] - 0.1, "$X_{i+1}$", fontsize=11, va="top")
    ax.text(
        0.8,
        -1.9,
        r"$X_{i+1} = \mathrm{Exp}_{X_i}(\hat V)$ (orange geodesic), $d_{AI}(X_i, X_{i+1}) = \|\hat V\|_{X_i} = 1$;"
        + "\nellipses: the exact points of the geodesic at t = 0, ¼, ½, ¾, 1 (2 × 2 example)",
        ha="center",
        fontsize=8.5,
        color=C["ink"],
    )
    # inset: whitened view, X_i -> I
    ins = ax.inset_axes([0.75, 0.45, 0.26, 0.56])
    ins.set_facecolor(C["bg"])
    s = np.linspace(0, 2 * np.pi, 200)
    ins.plot(np.cos(s), np.sin(s), color=C["green"], lw=1.5)
    ins.add_patch(
        FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="-|>",
            mutation_scale=11,
            color=C["green"],
            lw=1.8,
        )
    )
    ins.text(0.5, 0.1, r"$\hat W$", color=C["green"], fontsize=9, ha="center")
    for tk in (0, 0.5, 1.0):
        _spd_ellipse(
            ins,
            (1.1 * tk - 0.55, -1.55),
            _expm(tk * step["W_hat"]),
            C["ink"],
            lw=1.1,
            scale=0.17,
        )
    ins.text(
        0.05,
        1.12,
        r"whitened: $X_i \mapsto I$",
        ha="center",
        fontsize=8,
        color=C["ink"],
    )
    ins.text(
        0.05,
        -2.15,
        r"$\|\hat W\|_F = 1$; $e^{t\hat W}$, $t = 0, 1/2, 1$",
        ha="center",
        fontsize=7.5,
        color=C["muted"],
    )
    ins.set_xlim(-1.35, 1.45)
    ins.set_ylim(-2.35, 1.35)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_visible(True)
        spine.set_color(C["faint"])
    _save(fig, "rresnet_manifold")


def rresnet_le() -> None:
    """Log-Euclidean block: a translation in log coordinates."""
    log_x = _logm(X_I)
    q = _rot(Q_ANGLE)
    v = q @ np.diag(F_LAMBDA) @ q.T  # step "none": V as it is
    # exact coordinates in the plane spanned by log X_i and V (Frobenius-orthonormal
    # basis, drawn at 1.5 units per Frobenius unit); origin = log I = 0
    e1 = log_x / np.linalg.norm(log_x)
    e2 = v - np.sum(v * e1) * e1
    e2 /= np.linalg.norm(e2)
    k = 1.5

    def coords(sym):
        return k * np.sum(sym * e1), k * np.sum(sym * e2)

    a = coords(log_x)
    b = coords(log_x + v)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    xlim, ylim = (-0.6, a[0] + 2.4), (min(a[1], b[1]) - 0.9, max(a[1], b[1]) + 0.9)
    _frame(ax, (xlim[0], xlim[1]), (ylim[0] - 0.7, ylim[1] + 0.35))
    for g in np.arange(-0.5, xlim[1], 0.5):
        ax.plot([g, g], ylim, color=C["faint"], lw=0.7)
    for g in np.arange(np.ceil(ylim[0] * 2) / 2, ylim[1], 0.5):
        ax.plot(xlim, [g, g], color=C["faint"], lw=0.7)
    ax.plot(0, 0, "o", color=C["muted"], ms=4)
    ax.text(0.06, -0.08, r"$\log I = 0$", fontsize=8, color=C["muted"], va="top")
    _arrow(ax, a, b, color=C["green"], lw=2.2, ms=14)
    ax.text(
        a[0] - 0.12, (a[1] + b[1]) / 2, "$V$", color=C["green"], fontsize=11, ha="right"
    )
    for tk in (0, 0.5, 1.0):
        p = coords(log_x + tk * v)
        ax.plot(*p, "o", color=C["ink"], ms=4)
        ax.plot(
            [p[0] + 0.06, p[0] + 0.5], [p[1], p[1]], color=C["muted"], lw=0.7, ls=":"
        )
        _spd_ellipse(
            ax,
            (p[0] + 0.85, p[1]),
            _expm(log_x + tk * v),
            C["ink"],
            lw=1.3,
            scale=0.2,
            fill=C["bg"],
        )
    ax.text(a[0] + 1.3, a[1], r"$X_i$", fontsize=10, va="center")
    ax.text(b[0] + 1.3, b[1], r"$X_{i+1}$", fontsize=10, va="center")
    ax.text(a[0] - 0.1, a[1] - 0.12, r"$\log X_i$", ha="right", va="top", fontsize=10)
    ax.text(b[0] - 0.1, b[1] + 0.1, r"$\log X_i + V$", ha="right", fontsize=10)
    ax.text(
        np.mean(xlim),
        ylim[1] + 0.1,
        r"$\mathrm{Sym}_n$ in log coordinates (flat): $X_{i+1} = \exp(\log X_i + V)$",
        ha="center",
        fontsize=9,
    )
    ax.text(
        np.mean(xlim),
        ylim[0] - 0.65,
        r"a translation of $\log X$, $= \mathrm{Exp}^{LE}_{X_i}(V')$ with $V' = D\exp_{\log X_i}[V]$;"
        + "\n"
        + r"le_output='log' returns $\log X_i + V$ (no exp, no LogEig)",
        ha="center",
        fontsize=8.2,
        color=C["muted"],
    )
    _save(fig, "rresnet_le")


def rresnet_variants() -> None:
    """Res2SPDNet-inspired options: branch with parallel transport, congruence."""
    x = X_I
    z = _spd((1.3, 0.6), -20)  # the branch output Z = F(X) (BiMap + ReEig)
    q = _rot(Q_ANGLE)
    vz = q @ np.diag(F_LAMBDA) @ q.T  # field computed at Z
    xs, xis = _powm(x, 0.5), _powm(x, -0.5)
    e = xs @ _powm(xis @ z @ xis, -0.5) @ xis  # Gamma_{Z->X}(V) = E V E^T
    transported = e @ vz @ e.T
    zis = _powm(z, -0.5)
    norm_z = np.linalg.norm(zis @ vz @ zis)
    assert abs(np.linalg.norm(xis @ transported @ xis) - norm_z) < 1e-10  # isometry
    x_next = xs @ _expm(xis @ (transported / norm_z) @ xis) @ xs
    cong = xs @ z @ xs
    assert np.allclose(cong, xs @ _expm(_logm(z)) @ xs)
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    ax = axes[0]
    _frame(ax, (-0.3, 6.2), (-1.1, 2.6))
    _spd_ellipse(ax, (0.6, 0.6), x, C["ink"], lw=1.6, scale=0.32)
    _spd_ellipse(ax, (3.0, 1.75), z, C["purple"], lw=1.6, scale=0.32)
    _spd_ellipse(ax, (5.5, 0.6), x_next, C["ink"], lw=1.6, scale=0.32)
    ax.text(0.6, -0.2, "$X$", ha="center", fontsize=10)
    ax.text(
        3.75,
        1.75,
        "$Z = F(X)$\n(BiMap + ReEig)",
        ha="left",
        va="center",
        fontsize=8.5,
        color=C["purple"],
    )
    ax.text(5.5, -0.2, "$X_+$", ha="center", fontsize=10)
    _arrow(ax, (0.95, 1.0), (2.6, 1.75), color=C["purple"], rad=-0.15)
    _sym_cross(ax, (3.0, 0.95), vz, scale=0.3, lw=1.6)
    ax.text(3.4, 0.95, "$V(Z)$", fontsize=8.5, va="center")
    _sym_cross(ax, (1.9, -0.25), transported / norm_z, scale=0.45, lw=1.6)
    _arrow(ax, (2.75, 0.7), (2.15, 0.05), color=C["muted"], rad=0.15)
    ax.text(2.6, 0.2, r"$\Gamma_{Z\to X}$", fontsize=8.5, color=C["muted"])
    ax.text(
        1.9,
        -0.95,
        r"$\hat V = \Gamma_{Z\to X}V(Z) / \|V(Z)\|_Z$",
        ha="center",
        fontsize=8.2,
    )
    _arrow(ax, (1.05, 0.55), (5.05, 0.55), color=C["green"], lw=1.6)
    ax.text(
        4.3,
        0.25,
        r"$\mathrm{Exp}_X(\hat V)$",
        fontsize=9,
        color=C["green"],
        ha="center",
    )
    ax.set_title("residual='branch_pt'", fontsize=9.5)
    ax = axes[1]
    _frame(ax, (-0.3, 4.6), (-1.1, 2.6))
    _spd_ellipse(ax, (0.5, 1.0), x, C["ink"], lw=1.6, scale=0.32)
    _spd_ellipse(ax, (2.0, 1.0), z, C["purple"], lw=1.6, scale=0.32)
    _spd_ellipse(ax, (3.75, 1.0), cong, C["ink"], lw=1.6, scale=0.32)
    ax.text(0.5, 0.2, "$X$", ha="center", fontsize=10)
    ax.text(2.0, 0.2, "$Z = F(X)$", ha="center", fontsize=9, color=C["purple"])
    ax.text(3.75, 0.2, r"$X^{1/2} Z X^{1/2}$", ha="center", fontsize=9)
    ax.text(1.25, 0.95, "⊕", ha="center", fontsize=15, color=C["muted"])
    ax.text(2.9, 0.95, "=", ha="center", fontsize=14, color=C["muted"])
    ax.text(
        2.15,
        -0.6,
        r"$= \mathrm{Exp}_X(\Gamma_{I\to X} \log Z)$: no field, no step",
        ha="center",
        fontsize=8.5,
        color=C["muted"],
    )
    ax.set_title("residual='congruence'", fontsize=9.5)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "rresnet_variants")


DIAGRAMS = (
    daleckii_krein,
    implicit_diff,
    parametrization,
    reeig,
    spdnet_pipeline,
    package_levels,
    batchnorm_steps,
    gbwbn_steps,
    gbwbn_folding,
    rresnet_pipeline,
    rresnet_spectrum,
    rresnet_manifold,
    rresnet_le,
    rresnet_variants,
)


def main() -> None:
    global PNG_DIR
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--png", type=Path, help="also write PNG previews in this folder"
    )
    parser.add_argument("names", nargs="*", help="only these diagrams")
    args = parser.parse_args()
    PNG_DIR = args.png
    for theme in THEMES:
        _use(theme)
        for diagram in DIAGRAMS:
            if not args.names or diagram.__name__ in args.names:
                diagram()
    print("written to", OUT)


if __name__ == "__main__":
    main()
