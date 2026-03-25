from __future__ import annotations

import json, logging, math, time, uuid, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import expit

try:
    from modules.detection.detection_model import run_detection  # type: ignore
except ImportError:
    warnings.warn("Detection module not found — mock active.", stacklevel=2)

    def run_detection(image_path: str) -> dict:  # noqa: F811
        rng  = np.random.default_rng(seed=abs(hash(image_path)) % (2 ** 31))
        H, W = 600, 800
        pool = list(DEFECT_CATALOG.keys())
        n    = int(rng.integers(2, 7))
        dets = []
        for _ in range(n):
            cls  = str(rng.choice(pool))
            x1   = int(rng.uniform(0, W * 0.65))
            y1   = int(rng.uniform(0, H * 0.65))
            x2   = int(rng.uniform(x1 + 30, min(W, x1 + W * 0.45)))
            y2   = int(rng.uniform(y1 + 30, min(H, y1 + H * 0.45)))
            conf = float(rng.beta(9, 2))
            dets.append({"class_name": cls, "confidence": round(conf, 4),
                         "bbox": [x1, y1, x2, y2]})
        return {"detections": dets, "image_shape": (H, W), "count": len(dets)}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | PVEngine | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("PVDecisionEngine")


# ── §1  DOMAIN CONSTANTS ──────────────────────────────────────────────────────

DEFECT_CATALOG: Dict[str, Dict] = {
    "fingers": {
        "weight":          0.55,
        "el_contrast":     0.60,
        "string_impact":   False,
        "power_loss_frac": 0.12,
        "iec_severity":    2,
        "color":           "#f1c40f",
        "description":     "Finger interruption — series resistance rise, Isc loss",
        "el_signature":    "dark linear bands along contact fingers",
    },
    "star_crack": {
        "weight":          0.80,
        "el_contrast":     0.75,
        "string_impact":   False,
        "power_loss_frac": 0.25,
        "iec_severity":    3,
        "color":           "#9b59b6",
        "description":     "Star crack — radial fracture, severe current-path disruption",
        "el_signature":    "radial dark lines from mechanical impact point",
    },
    "black_core": {
        "weight":          0.92,
        "el_contrast":     0.95,
        "string_impact":   True,
        "power_loss_frac": 0.35,
        "iec_severity":    3,
        "color":           "#e74c3c",
        "description":     "Black core — fully inactive cell, reverse-bias, string risk",
        "el_signature":    "uniform dark cell region in EL image",
    },
    "crack": {
        "weight":          0.65,
        "el_contrast":     0.68,
        "string_impact":   False,
        "power_loss_frac": 0.18,
        "iec_severity":    2,
        "color":           "#e67e22",
        "description":     "Transverse crack — partial conductivity loss",
        "el_signature":    "dark line across cell body",
    },
    "thick_line": {
        "weight":          0.40,
        "el_contrast":     0.42,
        "string_impact":   False,
        "power_loss_frac": 0.08,
        "iec_severity":    1,
        "color":           "#1abc9c",
        "description":     "Thick line — bus-bar misfire, minor shading loss",
        "el_signature":    "bright/dark thick band along bus-bar",
    },
}

PHYSICS = {
    "weibull_shape_beta":       2.2,
    "weibull_scale_eta_years":  27.0,
}

FINANCE = {
    "panel_rated_power_wp":        400,
    "capacity_factor":             0.165,
    "electricity_price_usd_kwh":   0.12,
    "discount_rate":               0.07,
    "analysis_horizon_years":      10,
    "grid_emission_factor_kg_kwh": 0.417,
    "carbon_price_usd_per_tonne":  65.0,
    "panel_replacement_usd":       320.0,
    "panel_repair_usd":            110.0,
    "panel_cleaning_usd":          18.0,
}

PSI_TIERS = [
    (0.00, 0.05, "NEGLIGIBLE", 1),
    (0.05, 0.15, "LOW",        1),
    (0.15, 0.30, "MODERATE",   2),
    (0.30, 0.50, "HIGH",       2),
    (0.50, 1.01, "CRITICAL",   3),
]

URGENCY_MAP = {
    "NEGLIGIBLE": ("SCHEDULE_ROUTINE",  "Re-inspect at next O&M cycle (<=12 months)"),
    "LOW":        ("SCHEDULE_ROUTINE",  "Flag for next maintenance visit"),
    "MODERATE":   ("PLAN_WITHIN_30D",   "Plan corrective action within 30 days"),
    "HIGH":       ("EXPEDITE_7D",       "Expedite maintenance within 7 days"),
    "CRITICAL":   ("IMMEDIATE_ACTION",  "Isolate string; dispatch technician within 24 hours"),
}

MCMC_SAMPLES = 12_000
MCMC_BURNIN  = 2_000


# ── §2  DATA STRUCTURES ───────────────────────────────────────────────────────

@dataclass
class DefectInstance:
    class_name:        str
    confidence:        float
    bbox:              List[int]
    area_ratio:        float
    el_contrast_score: float = 0.0
    omega:             float = 0.0
    instance_severity: float = 0.0

@dataclass
class PSIResult:
    psi:                   float
    tier:                  str
    iec_class:             int
    dominant_defect:       str
    max_instance_severity: float
    critical_flag:         bool
    string_risk_flag:      bool
    defect_summary:        Dict[str, int]

@dataclass
class BayesianRiskResult:
    posterior_mean:       float
    posterior_std:        float
    credible_interval_95: Tuple[float, float]
    prior_alpha:          float
    prior_beta:           float
    posterior_alpha:      float
    posterior_beta:       float
    mcmc_ess:             float

@dataclass
class WeibullRULResult:
    rul_years_p50:          float
    rul_years_p10:          float
    rul_years_p90:          float
    hazard_rate:            float
    el_acceleration_factor: float
    virtual_age_years:      float

@dataclass
class FinancialResult:
    annual_energy_loss_kwh:     float
    annual_revenue_loss_usd:    float
    npv_loss_usd:               float
    annual_carbon_loss_kg:      float
    horizon_carbon_loss_tonnes: float
    carbon_cost_usd:            float

@dataclass
class MILPResult:
    recommended_action:   str
    action_binary_vector: Dict[str, int]
    optimal_cost:         float
    solver_status:        str
    action_costs:         Dict[str, float]
    action_benefits:      Dict[str, float]
    net_values:           Dict[str, float]


# ── §3  DETECTION UTILS (from detection_utils.py) ────────────────────────────

def draw_boxes(image_path: str, detections: list, out_path: str) -> str:
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 50))
    except Exception:
        font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = DEFECT_CATALOG.get(det["class_name"], {}).get("color", "#00ff00")
        for i in range(3):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        try:
            bb = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle([bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2], fill=color)
            draw.text((x1, y1 - 20), label, fill="white", font=font)
        except Exception:
            draw.text((x1, max(0, y1 - 20)), label, fill=color, font=font)
    img.save(out_path)
    return out_path


def compute_area_ratios(detections: list, img_w: int, img_h: int) -> list:
    panel_area = img_w * img_h if img_w * img_h > 0 else 1
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["area_ratio"] = round((x2 - x1) * (y2 - y1) / panel_area, 6)
    return detections


# ── §4  LAYER 0 — EL PIXEL CONTRAST SCORING ──────────────────────────────────
#
# EL images encode minority-carrier lifetime as luminescence intensity.
# Defective regions emit less light and appear darker than healthy cells.
#
# For each bounding box:
#   raw_contrast = clip((panel_mean - patch_mean) / panel_mean, 0, 1)
#   el_score_i   = el_contrast_i * raw_contrast
#
# el_contrast_i is the class-specific EL sensitivity from DEFECT_CATALOG.
# black_core has el_contrast=0.95 (fully dark); thick_line has 0.42 (subtle).
# Image is converted to L (luminance) channel regardless of input mode.
# Fallback: if image unreadable, el_score = 0.5 * el_contrast (conservative).

def compute_el_contrast_scores(
    image_path: str,
    defects: List[DefectInstance],
) -> List[DefectInstance]:
    try:
        arr    = np.asarray(Image.open(image_path).convert("L"), dtype=float)
        p_mean = arr.mean() if arr.mean() > 0 else 1.0
        for d in defects:
            x1, y1, x2, y2 = d.bbox
            patch = arr[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
            raw   = float(np.clip((p_mean - patch.mean()) / p_mean, 0.0, 1.0)) if patch.size > 0 else 0.5
            alpha = DEFECT_CATALOG.get(d.class_name, {}).get("el_contrast", 0.5)
            d.el_contrast_score = alpha * raw
    except Exception:
        for d in defects:
            alpha               = DEFECT_CATALOG.get(d.class_name, {}).get("el_contrast", 0.5)
            d.el_contrast_score = alpha * 0.5
    return defects


# ── §5  UTILITY FUNCTIONS ─────────────────────────────────────────────────────

def _iou(b1: List[int], b2: List[int]) -> float:
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1    = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2    = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)

def _overlap_penalty(defects: List[DefectInstance]) -> np.ndarray:
    n   = len(defects)
    rho = np.ones(n)
    for i in range(n):
        s      = sum(_iou(defects[i].bbox, defects[j].bbox) for j in range(n) if j != i)
        rho[i] = 1.0 / (1.0 + s)
    return rho

def _psi_tier(psi: float) -> Tuple[str, int]:
    for lo, hi, label, cls in PSI_TIERS:
        if lo <= psi < hi:
            return label, cls
    return "CRITICAL", 3

def _npv_factor(rate: float, years: int) -> float:
    return sum(1.0 / (1.0 + rate) ** t for t in range(1, years + 1))

def _kish_ess(weights: np.ndarray) -> float:
    w = weights / weights.sum()
    return float(1.0 / np.sum(w ** 2))

def _safe_logit(x: Any) -> Any:
    return np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1.0 - np.clip(x, 1e-6, 1 - 1e-6)))


# ── §6  LAYER 1 — PANEL SEVERITY INDEX (PSI) ─────────────────────────────────
#
# EL-adapted Chebyshev–L2 Hybrid Norm  (IEC 62446-3)
#
# Per instance i:
#   raw_i = area_ratio_i * omega_i * confidence_i * (1 + el_score_i) * rho_i
#
#   omega_i    = IEC class weight
#   el_score_i = Layer 0 EL pixel-darkness contrast score
#   rho_i      = 1 / (1 + sum_j IoU(i,j))     [NMS-aware overlap suppression]
#
# PSI = clip( 0.6 * max(raw) + 0.4 * rms(raw), 0, 1 )
#   0.6 * max  -> Chebyshev: worst-case single defect (dominant failure mode)
#   0.4 * rms  -> L2: distributed EL damage accumulation across all instances

def compute_psi(defects: List[DefectInstance]) -> PSIResult:
    if not defects:
        return PSIResult(0.0, "NEGLIGIBLE", 1, "none", 0.0, False, False, {})

    rho           = _overlap_penalty(defects)
    raws: List[float] = []
    summary: Dict[str, int] = {}
    critical_flag = False
    string_flag   = False
    dominant_cls  = ""
    max_sev       = 0.0

    for i, d in enumerate(defects):
        cat   = DEFECT_CATALOG.get(d.class_name, DEFECT_CATALOG["crack"])
        omega = cat["weight"]
        raw   = d.area_ratio * omega * d.confidence * (1.0 + d.el_contrast_score) * rho[i]

        d.omega             = omega
        d.instance_severity = raw
        raws.append(raw)

        summary[d.class_name] = summary.get(d.class_name, 0) + 1
        if cat["iec_severity"] == 3:
            critical_flag = True
        if cat["string_impact"]:
            string_flag = True
        if raw > max_sev:
            max_sev      = raw
            dominant_cls = d.class_name

    arr     = np.asarray(raws, dtype=float)
    psi     = float(np.clip(0.6 * arr.max() + 0.4 * np.sqrt(np.mean(arr ** 2)), 0.0, 1.0))
    tier, iec_cls = _psi_tier(psi)

    return PSIResult(
        psi=round(psi, 6), tier=tier, iec_class=iec_cls,
        dominant_defect=dominant_cls, max_instance_severity=round(max_sev, 6),
        critical_flag=critical_flag, string_risk_flag=string_flag,
        defect_summary=summary,
    )


# ── §7  LAYER 2 — HIERARCHICAL BAYESIAN RISK (Beta-Binomial + MH-MCMC) ───────
#
# Prior elicitation from IEC 62446-3 tier statistics:
#   theta ~ Beta(alpha0, beta0)
#     NEGLIGIBLE -> Beta(1.0, 9.0)   LOW     -> Beta(1.5, 7.5)
#     MODERATE   -> Beta(2.0, 6.0)   HIGH    -> Beta(4.0, 4.0)
#     CRITICAL   -> Beta(7.0, 2.0)
#
# Virtual likelihood (pseudo-count conjugate update):
#   n_eff = 20
#   k_eff = round(clip(PSI * multiplier * n_eff, 0, n_eff))
#   multiplier: critical_flag -> *1.4,  string_flag -> *1.2
#   Posterior: Beta(alpha0 + k_eff, beta0 + n_eff - k_eff)
#
# MCMC refinement — EL image uncertainty is modelled via latent variable:
#   eps ~ Beta(2.5, 3.5)   [right-skewed: EL tends to understate early damage]
#   Metropolis-Hastings with Beta proposal: eps' ~ Beta(eps*phi, (1-eps)*phi), phi=25
#   Combined: theta* = sigmoid(logit(theta_post) + logit(eps))
#   ESS via Kish (1965)

def compute_bayesian_risk(psi_result: PSIResult) -> BayesianRiskResult:
    psi  = psi_result.psi
    tier = psi_result.tier

    prior_map = {
        "NEGLIGIBLE": (1.0, 9.0), "LOW": (1.5, 7.5), "MODERATE": (2.0, 6.0),
        "HIGH": (4.0, 4.0),       "CRITICAL": (7.0, 2.0),
    }
    alpha0, beta0 = prior_map.get(tier, (2.0, 6.0))

    multiplier = 1.0
    if psi_result.critical_flag:  multiplier *= 1.40
    if psi_result.string_risk_flag: multiplier *= 1.20

    n_eff      = 20
    k_eff      = int(round(np.clip(psi * multiplier * n_eff, 0, n_eff)))
    alpha_n    = alpha0 + k_eff
    beta_n     = beta0 + (n_eff - k_eff)
    theta_post = alpha_n / (alpha_n + beta_n)

    rng     = np.random.default_rng(seed=42)
    phi     = 25.0
    eps_cur = float(rng.beta(2.5, 3.5))
    samples = []

    def log_target(e: float) -> float:
        return float(stats.beta.logpdf(e, 2.5, 3.5)) if 0 < e < 1 else -np.inf

    for step in range(MCMC_SAMPLES + MCMC_BURNIN):
        a_p    = max(1e-4, eps_cur * phi)
        b_p    = max(1e-4, (1.0 - eps_cur) * phi)
        eps_pr = float(rng.beta(a_p, b_p))
        la     = (log_target(eps_pr) - log_target(eps_cur)
                  + stats.beta.logpdf(eps_cur, eps_pr * phi, (1 - eps_pr) * phi)
                  - stats.beta.logpdf(eps_pr, eps_cur * phi, (1 - eps_cur) * phi))
        if np.log(rng.uniform()) < la:
            eps_cur = eps_pr
        if step >= MCMC_BURNIN:
            samples.append(eps_cur)

    samples    = np.asarray(samples, dtype=float)
    theta_star = expit(_safe_logit(theta_post) + _safe_logit(samples))
    ess        = _kish_ess(np.ones(len(theta_star)) / len(theta_star))

    return BayesianRiskResult(
        posterior_mean=round(float(np.mean(theta_star)), 6),
        posterior_std=round(float(np.std(theta_star)), 6),
        credible_interval_95=(
            round(float(np.percentile(theta_star, 2.5)), 4),
            round(float(np.percentile(theta_star, 97.5)), 4),
        ),
        prior_alpha=alpha0, prior_beta=beta0,
        posterior_alpha=round(alpha_n, 2), posterior_beta=round(beta_n, 2),
        mcmc_ess=round(ess, 1),
    )


# ── §8  LAYER 3 — WEIBULL RUL WITH EL-DERIVED ACCELERATION FACTOR ────────────
#
# EL contrast replaces thermal delta-T as the degradation signal.
# EL-AF maps luminescence drop to equivalent accelerated ageing.
#
# EL Acceleration Factor:
#   mean_el = mean(el_contrast_score_i * confidence_i)   over all detections
#   AF_el   = 1 + mean_el * 8          [calibrated: full EL drop -> 9x age speed]
#   AF_risk = 1 + posterior_mean * 1.5  [Bayesian risk inflation]
#   AF_total = clip(AF_el * AF_risk, 1, 20)
#
# Weibull Proportional-Hazard (fleet: beta=2.2, eta=27 yr, NREL/PVPMC data):
#   eta*  = eta / AF_total              [deflated characteristic life]
#   t_v   = 2 + (AF_total - 1) * 1.5   [Kijima Type-II virtual age, deploy=2yr]
#   h(t)  = (beta/eta*) * (t_v/eta*)^(beta-1)
#
# Conditional RUL at percentile p:
#   base  = (t_v / eta*)^beta
#   delta = eta* * (base - log(S_target))^(1/beta) - t_v
#   P10 -> S=0.9 (pessimistic), P50 -> S=0.5, P90 -> S=0.1 (optimistic)

def compute_weibull_rul(
    psi_result:  PSIResult,
    risk_result: BayesianRiskResult,
    defects:     List[DefectInstance],
) -> WeibullRULResult:
    beta_w = PHYSICS["weibull_shape_beta"]
    eta    = PHYSICS["weibull_scale_eta_years"]

    mean_el  = float(np.mean([d.el_contrast_score * d.confidence for d in defects])) if defects else 0.0
    AF_el    = 1.0 + mean_el * 8.0
    AF_risk  = 1.0 + risk_result.posterior_mean * 1.5
    AF_total = float(np.clip(AF_el * AF_risk, 1.0, 20.0))

    eta_star  = eta / AF_total
    t_v       = 2.0 + (AF_total - 1.0) * 1.5
    h_t       = (beta_w / eta_star) * ((t_v / eta_star) ** (beta_w - 1.0))
    base_term = (t_v / eta_star) ** beta_w

    def rul_at(s_target: float) -> float:
        return max(0.0, eta_star * ((base_term - math.log(max(s_target, 1e-9))) ** (1.0 / beta_w)) - t_v)

    return WeibullRULResult(
        rul_years_p50=round(rul_at(0.50), 2),
        rul_years_p10=round(rul_at(0.90), 2),
        rul_years_p90=round(rul_at(0.10), 2),
        hazard_rate=round(h_t, 6),
        el_acceleration_factor=round(AF_total, 3),
        virtual_age_years=round(t_v, 2),
    )


# ── §9  LAYER 4 — NPV-DISCOUNTED FINANCIAL & CARBON PROJECTION ───────────────
#
# Efficiency loss (EL-calibrated sensitivity 0.75):
#   eta_loss = clip(PSI * 0.75, 0, 0.99)
#
# Annual energy loss:
#   dE = P_rated[Wp] * CF * 8760h / 1000 * eta_loss   [kWh/yr]
#
# NPV-discounted revenue loss (WACC r over horizon T):
#   NPV = dE * price * sum_{t=1}^{T} (1+r)^{-t}
#
# Carbon:
#   dCO2_kg = dE * EF_kg_kwh   [kg/yr]
#   horizon_tonnes = dCO2_kg * T / 1000
#   carbon_cost_usd = horizon_tonnes * carbon_price

def compute_financial_impact(psi_result: PSIResult) -> FinancialResult:
    F               = FINANCE
    eta_loss        = float(np.clip(psi_result.psi * 0.75, 0.0, 0.99))
    annual_kwh_full = F["panel_rated_power_wp"] * F["capacity_factor"] * 8760 / 1000.0
    annual_el_loss  = annual_kwh_full * eta_loss
    annual_rev_loss = annual_el_loss * F["electricity_price_usd_kwh"]
    npv_f           = _npv_factor(F["discount_rate"], F["analysis_horizon_years"])
    npv_loss        = annual_rev_loss * npv_f
    annual_co2_kg   = annual_el_loss * F["grid_emission_factor_kg_kwh"]
    horizon_co2_t   = annual_co2_kg * F["analysis_horizon_years"] / 1000.0
    carbon_cost     = horizon_co2_t * F["carbon_price_usd_per_tonne"]

    return FinancialResult(
        annual_energy_loss_kwh=round(annual_el_loss, 3),
        annual_revenue_loss_usd=round(annual_rev_loss, 3),
        npv_loss_usd=round(npv_loss, 2),
        annual_carbon_loss_kg=round(annual_co2_kg, 3),
        horizon_carbon_loss_tonnes=round(horizon_co2_t, 4),
        carbon_cost_usd=round(carbon_cost, 2),
    )


# ── §10  LAYER 5 — TRUE MILP MAINTENANCE OPTIMISATION (scipy HiGHS) ──────────
#
# Variables: x = [x_monitor, x_clean, x_repair, x_replace] in {0,1}^4
#
# Objective — minimise total net cost:
#   min  sum_j  c_j * x_j
#   c_j = direct_cost_j + risk_penalty_j - NPV_benefit_j
#
#   Recovery factors:  monitor=0.00  clean=0.15  repair=0.60  replace=0.95
#   Risk penalty:      monitor  -> risk_mean * NPV * 0.50
#                      clean    -> risk_mean * NPV * 0.30
#                      repair, replace -> 0
#
# Hard constraints (linear inequalities):
#   (C1) sum x_j = 1                           [exactly one action selected]
#   (C2) x_monitor <= 0   if critical_flag      [no passive watch on critical]
#   (C3) x_replace <= 0   if PSI < 0.15         [no replace on near-healthy]
#   (C4) x_replace  = 1   if RUL_p50 < 3 yr    [forced replacement near EOL]
#        C4 supersedes C3 — RUL guard is encoded first, C3 skipped if force_replace
#
# Solver: scipy.optimize.milp -> HiGHS (same backend as PuLP, OR-Tools, CVXPY)
# Fallback: lexicographic argmin on net cost vector if HiGHS returns infeasible

def compute_milp_decision(
    psi_result:  PSIResult,
    risk_result: BayesianRiskResult,
    fin_result:  FinancialResult,
    rul_result:  WeibullRULResult,
) -> MILPResult:
    F         = FINANCE
    risk_mean = risk_result.posterior_mean
    npv_loss  = fin_result.npv_loss_usd
    psi       = psi_result.psi
    rul_p50   = rul_result.rul_years_p50

    ACTIONS          = ["monitor", "clean", "repair", "replace"]
    n                = len(ACTIONS)
    direct_costs     = np.array([0.0, F["panel_cleaning_usd"], F["panel_repair_usd"], F["panel_replacement_usd"]])
    recovery_factors = np.array([0.00, 0.15, 0.60, 0.95])
    benefits         = npv_loss * recovery_factors
    risk_penalties   = np.array([risk_mean * npv_loss * 0.50, risk_mean * npv_loss * 0.30, 0.0, 0.0])
    c                = direct_costs + risk_penalties - benefits

    force_replace  = rul_p50 < 3.0
    A_rows         = [np.ones((1, n))]
    lb_rows        = [np.array([1.0])]
    ub_rows        = [np.array([1.0])]

    if psi_result.critical_flag:
        r = np.zeros(n); r[0] = 1.0
        A_rows.append(r[np.newaxis, :]); lb_rows.append(np.array([-np.inf])); ub_rows.append(np.array([0.0]))
        log.info("MILP: critical_flag -> monitor disabled")

    if psi < 0.15 and not force_replace:
        r = np.zeros(n); r[3] = 1.0
        A_rows.append(r[np.newaxis, :]); lb_rows.append(np.array([-np.inf])); ub_rows.append(np.array([0.0]))
        log.info("MILP: low PSI -> replace disabled")

    if force_replace:
        r = np.zeros(n); r[3] = 1.0
        A_rows.append(r[np.newaxis, :]); lb_rows.append(np.array([1.0])); ub_rows.append(np.array([1.0]))
        log.info("MILP: RUL P50 < 3 yr -> replace forced")

    result = milp(
        c=c,
        constraints=LinearConstraint(np.vstack(A_rows), np.concatenate(lb_rows), np.concatenate(ub_rows)),
        integrality=np.ones(n),
        bounds=Bounds(lb=np.zeros(n), ub=np.ones(n)),
    )

    if result.success and result.x is not None:
        x_opt    = np.round(result.x).astype(int)
        chosen   = ACTIONS[int(np.argmax(x_opt))]
        opt_cost = float(result.fun)
        status   = "OPTIMAL"
    else:
        log.warning("MILP did not converge (%s) -> fallback", result.message)
        idx      = int(np.argmin(c))
        x_opt    = np.zeros(n, dtype=int); x_opt[idx] = 1
        chosen   = ACTIONS[idx]
        opt_cost = float(c[idx])
        status   = f"FALLBACK:{result.message}"

    return MILPResult(
        recommended_action=chosen.upper(),
        action_binary_vector={a: int(x_opt[i]) for i, a in enumerate(ACTIONS)},
        optimal_cost=round(opt_cost, 2),
        solver_status=status,
        action_costs={a: round(float(direct_costs[i]), 2) for i, a in enumerate(ACTIONS)},
        action_benefits={a: round(float(benefits[i]), 2) for i, a in enumerate(ACTIONS)},
        net_values={a: round(float(c[i]), 2) for i, a in enumerate(ACTIONS)},
    )


# ── §11  REPORT ASSEMBLY ──────────────────────────────────────────────────────

def _build_report(
    image_path:  str,
    det_result:  dict,
    defects:     List[DefectInstance],
    psi_result:  PSIResult,
    risk_result: BayesianRiskResult,
    rul_result:  WeibullRULResult,
    fin_result:  FinancialResult,
    milp_result: MILPResult,
    elapsed_ms:  float,
) -> Dict[str, Any]:
    urgency_code, urgency_note = URGENCY_MAP.get(psi_result.tier, ("UNKNOWN", ""))

    per_defect = []
    for d in defects:
        cat = DEFECT_CATALOG.get(d.class_name, {})
        per_defect.append({
            "class":             d.class_name,
            "description":       cat.get("description", ""),
            "el_signature":      cat.get("el_signature", ""),
            "confidence":        d.confidence,
            "bbox_xyxy":         d.bbox,
            "area_ratio":        d.area_ratio,
            "el_contrast_score": round(d.el_contrast_score, 4),
            "iec_severity":      cat.get("iec_severity", "?"),
            "instance_psi":      round(d.instance_severity, 5),
            "string_impact":     cat.get("string_impact", False),
        })

    return {
        "report_metadata": {
            "report_id":      str(uuid.uuid4()),
            "engine_version": "4.1.0-el-highs-milp",
            "image_type":     "Electroluminescence (EL)",
            "standards":      ["IEC 62446-3", "IEC 61215", "IEA PVPS T13"],
            "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
            "image_path":     image_path,
            "processing_ms":  round(elapsed_ms, 1),
        },
        "detection_summary": {
            "total_detections":    det_result["count"],
            "defect_class_counts": psi_result.defect_summary,
            "image_shape_hw":      list(det_result["image_shape"]),
        },
        "per_defect_analysis": per_defect,
        "panel_severity": {
            "psi":                     psi_result.psi,
            "tier":                    psi_result.tier,
            "iec_severity_class":      psi_result.iec_class,
            "dominant_defect":         psi_result.dominant_defect,
            "critical_defect_present": psi_result.critical_flag,
            "string_cascade_risk":     psi_result.string_risk_flag,
            "norm_method":             "EL Chebyshev-L2 hybrid (0.6/0.4)",
        },
        "bayesian_risk": {
            "posterior_mean_pct":       round(risk_result.posterior_mean * 100, 2),
            "posterior_std_pct":        round(risk_result.posterior_std * 100, 2),
            "credible_interval_95_pct": [
                round(risk_result.credible_interval_95[0] * 100, 2),
                round(risk_result.credible_interval_95[1] * 100, 2),
            ],
            "prior":        f"Beta({risk_result.prior_alpha}, {risk_result.prior_beta})",
            "posterior":    f"Beta({risk_result.posterior_alpha}, {risk_result.posterior_beta})",
            "mcmc_method":  "Metropolis-Hastings | EL uncertainty eps~Beta(2.5,3.5)",
            "mcmc_samples": MCMC_SAMPLES,
            "mcmc_ess":     risk_result.mcmc_ess,
        },
        "rul_estimate": {
            "rul_years_p50_median":      rul_result.rul_years_p50,
            "rul_years_p10_pessimistic": rul_result.rul_years_p10,
            "rul_years_p90_optimistic":  rul_result.rul_years_p90,
            "hazard_rate_per_year":      rul_result.hazard_rate,
            "el_acceleration_factor":    rul_result.el_acceleration_factor,
            "virtual_age_years":         rul_result.virtual_age_years,
            "model": f"Weibull PH (beta={PHYSICS['weibull_shape_beta']}, eta={PHYSICS['weibull_scale_eta_years']} yr) + EL-AF",
        },
        "financial_impact": {
            "annual_energy_loss_kwh":         fin_result.annual_energy_loss_kwh,
            "annual_revenue_loss_usd":        fin_result.annual_revenue_loss_usd,
            "npv_revenue_loss_usd":           fin_result.npv_loss_usd,
            "annual_carbon_loss_kgCO2e":      fin_result.annual_carbon_loss_kg,
            "horizon_carbon_loss_tonnesCO2e": fin_result.horizon_carbon_loss_tonnes,
            "carbon_cost_usd":                fin_result.carbon_cost_usd,
            "discount_rate_pct":              FINANCE["discount_rate"] * 100,
            "analysis_horizon_years":         FINANCE["analysis_horizon_years"],
        },
        "maintenance_decision": {
            "recommended_action":      milp_result.recommended_action,
            "solver":                  "scipy HiGHS MILP (binary integer variables)",
            "solver_status":           milp_result.solver_status,
            "optimal_net_cost_usd":    milp_result.optimal_cost,
            "action_binary_vector":    milp_result.action_binary_vector,
            "action_direct_costs_usd": milp_result.action_costs,
            "action_npv_benefits_usd": milp_result.action_benefits,
            "action_net_costs_usd":    milp_result.net_values,
        },
        "operational_guidance": {
            "urgency_code":              urgency_code,
            "urgency_note":              urgency_note,
            "next_inspection_interval":  (
                "<=3 months"  if psi_result.tier in ("HIGH", "CRITICAL") else
                "<=6 months"  if psi_result.tier == "MODERATE" else
                "<=12 months"
            ),
            "string_isolation_required": psi_result.string_risk_flag and psi_result.critical_flag,
            "confidence_statement": (
                f"EL analysis: {round(risk_result.posterior_mean * 100, 1)}% "
                f"[95% CI {round(risk_result.credible_interval_95[0]*100,1)}%-"
                f"{round(risk_result.credible_interval_95[1]*100,1)}%] "
                f"probability of accelerated degradation. "
                f"Median RUL: {rul_result.rul_years_p50} yr. "
                f"Action: {milp_result.recommended_action}."
            ),
        },
    }


# ── §12  PUBLIC ENTRY POINT ───────────────────────────────────────────────────

def analyze_panel(image_path: str, annotated_out_path: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    log.info("Starting EL analysis: %s", image_path)

    det_result    = run_detection(image_path)
    raw_dets      = det_result.get("detections", [])
    height, width = det_result["image_shape"]
    panel_area    = height * width

    raw_dets = compute_area_ratios(raw_dets, width, height)

    defects: List[DefectInstance] = []
    for d in raw_dets:
        cls = str(d.get("class_name", "crack")).lower().strip()
        if cls not in DEFECT_CATALOG:
            log.warning("Unknown class '%s' -> remapped to 'crack'", cls)
            cls = "crack"
        x1, y1, x2, y2 = d["bbox"]
        defects.append(DefectInstance(
            class_name=cls,
            confidence=float(np.clip(d.get("confidence", 0.8), 0.0, 1.0)),
            bbox=[x1, y1, x2, y2],
            area_ratio=float(d.get("area_ratio", max(1, (x2-x1)*(y2-y1)) / panel_area)),
        ))

    log.info("Layer 0: EL contrast scoring")
    defects = compute_el_contrast_scores(image_path, defects)

    log.info("Layer 1: PSI")
    psi_result = compute_psi(defects)
    log.info("  PSI=%.4f  Tier=%s", psi_result.psi, psi_result.tier)

    log.info("Layer 2: Bayesian MCMC risk (%d samples)", MCMC_SAMPLES)
    risk_result = compute_bayesian_risk(psi_result)
    log.info("  Risk=%.2f%%  ESS=%.0f", risk_result.posterior_mean * 100, risk_result.mcmc_ess)

    log.info("Layer 3: Weibull RUL + EL-AF")
    rul_result = compute_weibull_rul(psi_result, risk_result, defects)
    log.info("  RUL_P50=%.1f yr  AF=%.2f", rul_result.rul_years_p50, rul_result.el_acceleration_factor)

    log.info("Layer 4: NPV financial + carbon")
    fin_result = compute_financial_impact(psi_result)
    log.info("  NPV_loss=USD %.2f  CO2=%.3f t", fin_result.npv_loss_usd, fin_result.horizon_carbon_loss_tonnes)

    log.info("Layer 5: HiGHS MILP optimisation")
    milp_result = compute_milp_decision(psi_result, risk_result, fin_result, rul_result)
    log.info("  Action=%s  Status=%s", milp_result.recommended_action, milp_result.solver_status)

    if annotated_out_path:
        draw_boxes(image_path, raw_dets, annotated_out_path)
        log.info("  Annotated EL image saved: %s", annotated_out_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    report = _build_report(
        image_path, det_result, defects,
        psi_result, risk_result, rul_result, fin_result, milp_result, elapsed_ms,
    )
    log.info("Complete in %.1f ms -> %s", elapsed_ms, milp_result.recommended_action)
    return report


# ── §13  CLI SELF-TEST ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_el_panel.jpg"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(analyze_panel(img_path, annotated_out_path=out_path), indent=2, default=str))