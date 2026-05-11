#!/usr/bin/env python3
"""Fit a photopeak in an ADC spectrum with a bin-integrated Gaussian."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf


SQRT2 = np.sqrt(2.0)


def read_adc_values(path: Path) -> np.ndarray:
    """Read one ADC value per line and discard non-finite entries."""
    values = np.loadtxt(path, dtype=float)
    values = np.asarray(values).reshape(-1)
    return values[np.isfinite(values)]


def make_histogram(
    values: np.ndarray,
    *,
    bin_width: float,
    adc_min: float,
    adc_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return histogram counts, bin edges, and bin centers."""
    edges = np.arange(adc_min, adc_max + bin_width, bin_width)
    counts, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), edges.astype(float), centers


def gaussian_bin_integral(
    x_edges: np.ndarray,
    area: float,
    center: float,
    sigma: float,
    background: float,
    slope: float,
    x_ref: float,
) -> np.ndarray:
    """Expected bin counts from a Gaussian peak plus a linear background.

    ``area`` is the total number of Gaussian peak counts.  The Gaussian part is
    integrated between each bin's left and right edges; this is the correct
    comparison target for histogram counts.
    """
    left, right = x_edges
    centers = 0.5 * (left + right)
    z_left = (left - center) / (SQRT2 * sigma)
    z_right = (right - center) / (SQRT2 * sigma)
    peak = 0.5 * area * (erf(z_right) - erf(z_left))
    baseline = background + slope * (centers - x_ref)
    return peak + baseline


def guess_initial_parameters(
    counts: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    peak_guess: float,
) -> tuple[float, float, float, float, float]:
    """Estimate initial fit parameters from the selected fit window."""
    edge_bins = max(2, len(counts) // 6)
    left_level = np.median(counts[:edge_bins])
    right_level = np.median(counts[-edge_bins:])
    x_ref = peak_guess

    left_x = np.mean(centers[:edge_bins])
    right_x = np.mean(centers[-edge_bins:])
    slope = (right_level - left_level) / max(right_x - left_x, 1.0)
    background = max(0.5 * (left_level + right_level), 0.0)

    baseline = background + slope * (centers - x_ref)
    signal = np.clip(counts - baseline, 0.0, None)
    if signal.sum() > 0:
        center = np.average(centers, weights=signal)
        variance = np.average((centers - center) ** 2, weights=signal)
        area = signal.sum()
    else:
        center = peak_guess
        variance = (0.15 * (centers[-1] - centers[0])) ** 2
        area = max(counts.sum() - background * len(counts), counts.max())

    bin_width = np.median(np.diff(edges))
    sigma = np.sqrt(max(variance, (1.5 * bin_width) ** 2))
    sigma = min(sigma, 0.5 * (edges[-1] - edges[0]))
    return area, center, sigma, background, slope


def fit_photopeak(
    counts: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    *,
    search_min: float,
    search_max: float,
    fit_half_width: float,
) -> dict[str, np.ndarray | float | int]:
    """Find and fit the strongest peak in the search interval."""
    search_mask = (centers >= search_min) & (centers <= search_max)
    if not np.any(search_mask):
        raise ValueError("no histogram bins are inside the peak search range")

    search_indices = np.flatnonzero(search_mask)
    peak_index = search_indices[np.argmax(counts[search_mask])]
    peak_guess = centers[peak_index]

    fit_mask = (
        (centers >= peak_guess - fit_half_width)
        & (centers <= peak_guess + fit_half_width)
    )
    if fit_mask.sum() < 8:
        raise ValueError("fit range is too narrow; increase --fit-half-width")

    y = counts[fit_mask]
    left = edges[:-1][fit_mask]
    right = edges[1:][fit_mask]
    x = centers[fit_mask]
    x_edges = np.vstack([left, right])
    yerr = np.sqrt(np.maximum(y, 1.0))
    x_ref = peak_guess

    p0 = guess_initial_parameters(y, np.r_[left, right[-1]], x, peak_guess)

    def model(x_edges_: np.ndarray, area: float, center: float, sigma: float,
              background: float, slope: float) -> np.ndarray:
        return gaussian_bin_integral(
            x_edges_, area, center, sigma, background, slope, x_ref
        )

    lower = [0.0, left[0], 0.5 * np.median(np.diff(edges)), 0.0, -np.inf]
    upper = [np.inf, right[-1], fit_half_width, np.inf, np.inf]
    popt, pcov = curve_fit(
        model,
        x_edges,
        y,
        p0=p0,
        sigma=yerr,
        absolute_sigma=True,
        bounds=(lower, upper),
        maxfev=20000,
    )

    expected = model(x_edges, *popt)
    chi2 = float(np.sum(((y - expected) / yerr) ** 2))
    ndf = int(len(y) - len(popt))
    return {
        "popt": popt,
        "pcov": pcov,
        "peak_guess": float(peak_guess),
        "fit_mask": fit_mask,
        "x_edges": x_edges,
        "x": x,
        "y": y,
        "yerr": yerr,
        "expected": expected,
        "chi2": chi2,
        "ndf": ndf,
        "x_ref": float(x_ref),
    }


def plot_fit(
    output: Path,
    *,
    title: str,
    counts: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    fit_result: dict[str, np.ndarray | float | int],
) -> None:
    """Save the histogram and fitted function as a PDF."""
    popt = np.asarray(fit_result["popt"])
    pcov = np.asarray(fit_result["pcov"])
    perr = np.sqrt(np.diag(pcov))
    fit_mask = np.asarray(fit_result["fit_mask"], dtype=bool)
    x_ref = float(fit_result["x_ref"])

    bin_width = np.median(np.diff(edges))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        centers,
        counts,
        width=bin_width,
        align="center",
        color="#7aa6c2",
        edgecolor="#335f7a",
        linewidth=0.35,
        alpha=0.65,
        label="histogram",
    )
    ax.errorbar(
        centers[fit_mask],
        counts[fit_mask],
        yerr=np.sqrt(np.maximum(counts[fit_mask], 1.0)),
        fmt="o",
        ms=3.0,
        color="#1f2933",
        ecolor="#1f2933",
        elinewidth=0.7,
        capsize=1.5,
        label="fit bins",
    )

    x_plot = np.linspace(edges[:-1][fit_mask][0], edges[1:][fit_mask][-1], 500)
    x_edges_plot = np.vstack([x_plot - 0.5 * bin_width, x_plot + 0.5 * bin_width])
    y_plot = gaussian_bin_integral(x_edges_plot, *popt, x_ref=x_ref)
    ax.plot(x_plot, y_plot, color="#b3261e", lw=2.0, label="bin-integrated fit")

    _area, center, sigma, _background, _slope = popt
    center_err = perr[1]
    sigma_err = perr[2]
    chi2 = float(fit_result["chi2"])
    ndf = int(fit_result["ndf"])
    annotation = (
        rf"$\mu = {center:.1f} \pm {center_err:.1f}$ ch" "\n"
        rf"$\sigma = {sigma:.1f} \pm {sigma_err:.1f}$ ch" "\n"
        rf"$\chi^2/\mathrm{{ndf}} = {chi2:.1f}/{ndf}$"
    )
    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#b8c2cc"},
    )

    ax.set_title(title)
    ax.set_xlabel("ADC channel")
    ax.set_ylabel(f"counts / {bin_width:g} ch")
    ax.set_xlim(edges[:-1][fit_mask][0] - 3 * bin_width, edges[1:][fit_mask][-1] + 3 * bin_width)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the photopeak of an ADC spectrum and save a PDF plot."
    )
    parser.add_argument("input", type=Path, help="text file with one ADC value per line")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output PDF path",
    )
    parser.add_argument("--bin-width", type=float, default=16.0, help="histogram bin width")
    parser.add_argument("--adc-min", type=float, default=0.0, help="minimum ADC channel")
    parser.add_argument("--adc-max", type=float, default=4096.0, help="maximum ADC channel")
    parser.add_argument(
        "--search-min",
        type=float,
        default=400.0,
        help="lower edge of the photopeak search region",
    )
    parser.add_argument(
        "--search-max",
        type=float,
        default=3900.0,
        help="upper edge of the photopeak search region",
    )
    parser.add_argument(
        "--fit-half-width",
        type=float,
        default=180.0,
        help="half-width of the fit window around the peak guess",
    )
    parser.add_argument("--title", default=None, help="plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = read_adc_values(args.input)
    counts, edges, centers = make_histogram(
        values,
        bin_width=args.bin_width,
        adc_min=args.adc_min,
        adc_max=args.adc_max,
    )
    fit_result = fit_photopeak(
        counts,
        edges,
        centers,
        search_min=args.search_min,
        search_max=args.search_max,
        fit_half_width=args.fit_half_width,
    )

    output = args.output
    if output is None:
        output = Path("figures") / f"{args.input.stem}_photopeak_fit.pdf"
    title = args.title or args.input.stem
    plot_fit(output, title=title, counts=counts, edges=edges, centers=centers,
             fit_result=fit_result)

    popt = np.asarray(fit_result["popt"])
    perr = np.sqrt(np.diag(np.asarray(fit_result["pcov"])))
    chi2 = float(fit_result["chi2"])
    ndf = int(fit_result["ndf"])
    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"area       = {popt[0]:.3g} +/- {perr[0]:.3g} counts")
    print(f"center     = {popt[1]:.3f} +/- {perr[1]:.3f} ch")
    print(f"sigma      = {popt[2]:.3f} +/- {perr[2]:.3f} ch")
    print(f"background = {popt[3]:.3f} +/- {perr[3]:.3f} counts/bin")
    print(f"slope      = {popt[4]:.4g} +/- {perr[4]:.4g} counts/bin/ch")
    print(f"chi2/ndf   = {chi2:.2f} / {ndf} = {chi2 / ndf:.2f}")


if __name__ == "__main__":
    main()
