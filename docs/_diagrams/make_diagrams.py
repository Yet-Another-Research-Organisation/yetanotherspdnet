"""Generate the SVG diagrams of the user guide (docs/_static/diagrams/).

Run from the repository root:  python docs/_diagrams/make_diagrams.py

Colors are chosen to read on both the light and the dark Furo themes
(transparent background, mid-tone lines and text).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "_static" / "diagrams"
INK = "#7a8194"  # neutral, legible on white and on dark grey
BLUE = "#4a64c8"
ORANGE = "#e07b39"
GREEN = "#3a9a5b"
RED = "#c94a4a"
plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "yetanotherspdnet",  # stable element ids: no diff noise
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{name}.svg",
        transparent=True,
        bbox_inches="tight",
        metadata={"Date": None},  # no timestamp: regenerating is a no-op
    )
    plt.close(fig)


def _arrow(ax, start, end, color=INK, style="-|>", lw=1.3, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _spd_ellipse(ax, center, matrix, color, alpha=0.9, scale=0.35, lw=1.4):
    """Draw the 2x2 SPD matrix as the ellipse {x : x^T M^{-1} x = 1}."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * scale * np.sqrt(eigvals[::-1])
    ax.add_patch(
        Ellipse(
            center,
            width,
            height,
            angle=angle,
            fill=False,
            color=color,
            lw=lw,
            alpha=alpha,
        )
    )


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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
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


def parametrization() -> None:
    """Static chart vs moving reference point on a curved manifold."""
    t = np.linspace(-0.2, np.pi + 0.2, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, title in zip(
        axes, ("static: one chart", "dynamic: chart re-centred"), strict=True
    ):
        ax.plot(np.cos(t), np.sin(t), color=INK, lw=1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    # static: tangent line at the initial point, far points badly represented
    p0 = np.array([np.cos(0.35), np.sin(0.35)])
    tangent = np.array([-np.sin(0.35), np.cos(0.35)])
    line = np.array([p0 + s * tangent for s in np.linspace(-0.4, 2.2, 2)])
    axes[0].plot(line[:, 0], line[:, 1], color=BLUE, lw=1.2, ls="--")
    axes[0].plot(*p0, "o", color=BLUE)
    axes[0].annotate(
        r"$W_0$", p0, xytext=(8, -12), textcoords="offset points", color=BLUE
    )
    far = np.array([np.cos(2.2), np.sin(2.2)])
    proj = p0 + ((far - p0) @ tangent) * tangent
    axes[0].plot(*far, "o", color=ORANGE)
    axes[0].plot([far[0], proj[0]], [far[1], proj[1]], color=ORANGE, lw=1, ls=":")
    axes[0].annotate(
        r"$W$ far from $W_0$:" + "\nchart distorted",
        far,
        xytext=(-60, 16),
        textcoords="offset points",
        color=ORANGE,
        fontsize=8,
    )
    # dynamic: a few reference points along the trajectory
    prev = None
    for k, a in enumerate((0.35, 1.05, 1.75, 2.4)):
        p = np.array([np.cos(a), np.sin(a)])
        tg = np.array([-np.sin(a), np.cos(a)])
        seg = np.array([p - 0.35 * tg, p + 0.35 * tg])
        axes[1].plot(seg[:, 0], seg[:, 1], color=BLUE, lw=1.2, ls="--")
        axes[1].plot(*p, "o", color=BLUE if k < 3 else ORANGE)
        if prev is not None:
            _arrow(axes[1], prev * 1.28, p * 1.28, color=GREEN, rad=0.25)
        prev = p
    axes[1].text(
        0.0,
        0.3,
        r"$W_{ref} \leftarrow W$" + "\nevery n steps",
        color=GREEN,
        fontsize=8,
        ha="center",
    )
    _save(fig, "parametrization")


def batchnorm_steps() -> None:
    """2x2 SPD matrices as ellipses: batch, centred, rescaled, biased."""
    rng = np.random.default_rng(3)
    base = np.array([[2.2, 0.9], [0.9, 0.8]])
    batch = []
    for _ in range(7):
        a = rng.normal(size=(2, 2)) * 0.35
        m = base + a @ a.T
        batch.append(m)
    batch = np.array(batch)
    # affine-invariant (Karcher) mean approximated by the log-Euclidean mean
    logs = np.array([_logm(m) for m in batch])
    mean = _expm(logs.mean(0))
    isq = _powm(mean, -0.5)
    centred = np.array([isq @ m @ isq for m in batch])
    spread = np.sqrt(
        np.mean([np.sum(np.log(np.linalg.eigvalsh(c)) ** 2) for c in centred])
    )
    s = 0.6
    scaled = np.array([_powm(c, s / spread) for c in centred])
    bias = np.array([[0.7, -0.35], [-0.35, 1.6]])
    bsq = _powm(bias, 0.5)
    biased = np.array([bsq @ c @ bsq for c in scaled])
    stages = [
        (batch, "batch $X_i$", mean, r"mean $\bar X$"),
        (centred, r"centred $\bar X^{-1/2}X_i\bar X^{-1/2}$", np.eye(2), "$I$"),
        (scaled, r"rescaled $(\cdot)^{s/\sigma}$", np.eye(2), "$I$"),
        (biased, r"biased $G^{1/2}(\cdot)G^{1/2}$", bias, "$G$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.6))
    for ax, (mats, title, ref, ref_label) in zip(axes, stages, strict=True):
        for m in mats:
            _spd_ellipse(ax, (0, 0), m, BLUE, alpha=0.6, lw=1.1)
        _spd_ellipse(ax, (0, 0), ref, ORANGE, alpha=1.0, lw=2.2)
        ax.text(0, -0.95, ref_label, color=ORANGE, ha="center", fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8.5)
    for i in range(3):
        fig.text(0.255 + 0.245 * i, 0.5, "→", fontsize=18, color=INK, ha="center")
    fig.tight_layout()
    _save(fig, "batchnorm_steps")


def reeig() -> None:
    lam = np.logspace(-4, 1, 400)
    eps = 1e-2
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.loglog(lam, lam, color=INK, lw=1, ls=":", label="identity")
    ax.loglog(
        lam,
        np.maximum(lam, eps),
        color=BLUE,
        lw=2,
        label=r"ReEig $\max(\lambda, \epsilon)$",
    )
    ax.loglog(
        lam,
        np.clip(lam + 0.03, eps, 1 / 0.2),
        color=ORANGE,
        lw=1.6,
        label=r"ReEigBias $\mathrm{clamp}(\lambda+b, \epsilon, 1/\epsilon)$",
    )
    ax.axvline(eps, color=INK, lw=0.8, ls="--")
    ax.text(eps * 1.2, 2e-4, r"$\epsilon$", color=INK)
    ax.set_xlabel(r"eigenvalue $\lambda$ of the input")
    ax.set_ylabel("output eigenvalue")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "reeig")


def retractions() -> None:
    x = np.linspace(-0.95, 0.95, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    curves = [
        (np.exp(x), r"AI: $e^x$", BLUE, "-"),
        (1 + x, r"arithmetic: $1+x$", ORANGE, "-"),
        (1 / (1 - x), r"harmonic: $1/(1-x)$", RED, "-"),
        (
            np.sqrt((1 + x) / (1 - x)),
            r"GAH: $e^{\operatorname{artanh} x}$",
            GREEN,
            "--",
        ),
    ]
    for y, label, color, ls in curves:
        axes[0].plot(x, y, color=color, ls=ls, lw=1.6, label=label)
    axes[0].set_ylim(0, 4)
    axes[0].axvspan(-1, -0.95, color=RED, alpha=0.15)
    axes[0].axvspan(0.95, 1, color=RED, alpha=0.15)
    axes[0].set_xlabel(r"eigenvalue $x$ of the normalized step $a\hat W$")
    axes[0].set_ylabel(r"$\varphi(x)$")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title(
        r"$X_+ = L\,\varphi(a\hat W)\,L^\top$, valid for $|x|<1$", fontsize=9
    )
    # unit step: every matrix moves by the same AI distance
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.4)
    t = np.linspace(0, 4, 100)
    ax.plot(t, 0.45 + 0.12 * np.sin(1.3 * t), color=INK, lw=1.4)
    for x0, v in ((0.6, 1.0), (2.1, 1.0), (3.2, 1.0)):
        y0 = 0.45 + 0.12 * np.sin(1.3 * x0)
        ax.plot(x0, y0, "o", color=BLUE)
        _arrow(ax, (x0, y0), (x0 + 0.35, y0 + v), color=ORANGE, rad=-0.25)
    ax.text(
        2.0,
        1.95,
        r"$X_+ = \mathrm{Exp}_X(V/\|V\|_X)$" + "\nunit affine-invariant step",
        ha="center",
        fontsize=8.5,
        color=ORANGE,
    )
    ax.text(
        2.0,
        0.05,
        r"$\|V\|_X = \|L^{-1} V L^{-\top}\|_F$, $X = LL^\top$",
        ha="center",
        fontsize=8,
        color=INK,
    )
    _save(fig, "retractions")


def bw_fold() -> None:
    v = np.linspace(-5, 2.5, 400)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(v, (1 + v / 2) ** 2, color=BLUE, lw=2)
    ax.axvspan(-5, -2, color=RED, alpha=0.12)
    ax.axvline(-2, color=RED, lw=1, ls="--")
    ax.text(-4.9, 5.0, "outside the injectivity\ndomain: folded", color=RED, fontsize=8)
    ax.text(-1.6, 5.0, r"$1 + v/2 > 0$: injective", color=BLUE, fontsize=8)
    ax.plot([-3, 1], [0.25, 2.25], "o", color=ORANGE, ms=4)
    ax.annotate(
        "",
        (1, 2.25),
        (-3, 0.25),
        arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 0.8, "ls": ":"},
    )
    ax.plot(-2, 0, "o", color=RED, ms=5)
    ax.annotate(
        "singular output",
        (-2, 0),
        xytext=(-1.6, -0.9),
        color=RED,
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.8},
    )
    ax.set_xlabel(r"eigenvalue $v$ of the transported tangent vector")
    ax.set_ylabel(r"$\mathrm{Exp}_I(v) = (1 + v/2)^2$")
    ax.set_ylim(-1.2, 6)
    _save(fig, "bw_fold")


def implicit_diff() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    boxes = [
        (0.3, "$\\Sigma_0$"),
        (1.9, "$\\Sigma_1$"),
        (3.5, "…"),
        (5.1, "$\\Sigma_K$"),
    ]
    for x, label in boxes:
        ax.text(
            x + 0.5,
            2.2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round", "fc": "none", "ec": BLUE},
        )
    for x in (0.3, 1.9, 3.5):
        _arrow(ax, (x + 0.85, 2.2), (x + 1.75, 2.2), color=BLUE)
    ax.text(
        3.1,
        2.75,
        r"forward: $\Sigma_{k+1} = F(\Sigma_k, x)$",
        ha="center",
        color=BLUE,
        fontsize=9,
    )
    ax.text(
        3.1,
        1.35,
        "autograd: backward through the K iterations\n(memory ∝ K)",
        ha="center",
        color=INK,
        fontsize=8.5,
    )
    ax.text(
        8.0,
        2.2,
        r"$\Sigma^\star = F(\Sigma^\star, x)$",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round", "fc": "none", "ec": GREEN},
    )
    _arrow(ax, (5.95, 2.2), (6.95, 2.2), color=GREEN)
    ax.text(
        8.0,
        1.2,
        "implicit: solve $w = g + J_\\Sigma^\\top w$ once,\n"
        "$\\bar x = J_x^\\top w$ (memory independent of K)",
        ha="center",
        color=GREEN,
        fontsize=8.5,
    )
    _save(fig, "implicit_diff")


def _box(ax, x, y, w, h, text, color, fontsize=8.5, fill_alpha=0.08):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color,
            ec=color,
            alpha=fill_alpha,
            lw=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc="none",
            ec=color,
            lw=1.3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK if color == INK else color,
    )


def spdnet_pipeline() -> None:
    """SPDNet on HDM05: layers and shapes, manifold part vs Euclidean head."""
    fig, ax = plt.subplots(figsize=(9.6, 2.5))
    ax.axis("off")
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 3)
    steps = [
        ("X", "(B, 93, 93)", INK),
        ("BiMap", "(B, 84, 84)", BLUE),
        ("ReEig", "(B, 84, 84)", BLUE),
        ("BatchNorm", "(B, 84, 84)", ORANGE),
        ("BiMap\nReEig\nBatchNorm", "(B, 63, 63)", BLUE),
        ("LogEig", "(B, 63, 63)", GREEN),
        ("Vec", "(B, 3969)", GREEN),
        ("Linear", "(B, 117)", GREEN),
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
            color=INK,
            family="monospace",
        )
        if i:
            _arrow(ax, (x - gap + 0.02, 1.7), (x - 0.02, 1.7), color=INK)
    x_log = 0.15 + 5 * (width + gap)
    ax.plot([0.15, x_log - 0.14], [2.5, 2.5], color=BLUE, lw=1.2)
    ax.text(
        (0.15 + x_log) / 2,
        2.62,
        "on the SPD manifold: every output is SPD",
        ha="center",
        color=BLUE,
        fontsize=8.5,
    )
    ax.plot([x_log, 12.05], [2.5, 2.5], color=GREEN, lw=1.2)
    ax.text(
        (x_log + 12.05) / 2,
        2.62,
        "flat space: Euclidean head",
        ha="center",
        color=GREEN,
        fontsize=8.5,
    )
    ax.text(
        6.1,
        0.25,
        "SPDnet(input_dim=93, hidden_layers_size=[84, 63], output_dim=117, "
        "batchnorm=True)",
        ha="center",
        fontsize=8,
        color=INK,
        family="monospace",
    )
    _save(fig, "spdnet_pipeline")


def package_levels() -> None:
    """The three levels of the package and the neighbouring repositories."""
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    levels = [
        (
            3.0,
            "model",
            "SPDnet · RResNet · GBWBNRResNet",
            "classifiers, one constructor",
            ORANGE,
        ),
        (
            1.7,
            "nn",
            "BiMap · ReEig · LogEig · BatchNorm · ResidualBlock ·\n"
            "SampleCovariance · MEstimation · parametrizations",
            "modules with parameters",
            BLUE,
        ),
        (
            0.3,
            "functions",
            "spd_linalg · spd_geometries/* · m_estimators · stiefel",
            "pure tensor functions, two gradient paths",
            GREEN,
        ),
    ]
    for y, name, content, role, color in levels:
        _box(ax, 2.2, y, 5.6, 1.0, "", color)
        ax.text(2.4, y + 0.72, name, fontsize=10, color=color, weight="bold")
        ax.text(
            5.0,
            y + 0.5,
            content,
            ha="center",
            va="center",
            fontsize=7.8,
            color=INK,
            family="monospace",
        )
        ax.text(
            7.65, y + 0.72, role, ha="right", fontsize=7.5, color=color, style="italic"
        )
    for y0, y1 in ((3.0, 2.7), (1.7, 1.3)):
        _arrow(ax, (5.0, y0 - 0.02), (5.0, y1 + 0.02), color=INK)
    _box(
        ax,
        0.05,
        1.7,
        1.75,
        1.0,
        "spdnet-datasets\n(loaders, SPD data)",
        INK,
        fontsize=7.5,
        fill_alpha=0.04,
    )
    _arrow(ax, (1.85, 2.2), (2.15, 2.2), color=INK)
    _box(
        ax,
        8.2,
        2.35,
        1.75,
        1.0,
        "spdnet-training\n(Lightning, Hydra,\nbackbones)",
        INK,
        fontsize=7.5,
        fill_alpha=0.04,
    )
    _box(
        ax,
        8.2,
        0.9,
        1.75,
        1.0,
        "benchmark demo,\npaper repositories",
        INK,
        fontsize=7.5,
        fill_alpha=0.04,
    )
    _arrow(ax, (8.15, 2.85), (7.85, 2.85), color=INK)
    _arrow(ax, (8.15, 1.4), (7.85, 1.4), color=INK)
    ax.text(
        5.0,
        4.05,
        "yetanotherspdnet",
        ha="center",
        fontsize=11,
        color=INK,
        weight="bold",
    )
    _save(fig, "package_levels")


def _eig_fun(m, f):
    w, u = np.linalg.eigh(m)
    return (u * f(w)) @ u.T


def _logm(m):
    return _eig_fun(m, np.log)


def _expm(m):
    return _eig_fun(m, np.exp)


def _powm(m, p):
    return _eig_fun(m, lambda w: w**p)


if __name__ == "__main__":
    daleckii_krein()
    parametrization()
    batchnorm_steps()
    reeig()
    retractions()
    bw_fold()
    implicit_diff()
    spdnet_pipeline()
    package_levels()
    print("written to", OUT)
