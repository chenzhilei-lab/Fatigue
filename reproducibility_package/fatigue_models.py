"""
fatigue_models.py — Five-switch fatigue life calculation framework
==================================================================
Each function = one switch. The main pipeline loops over all
combinations and outputs fatigue life Nf (cycles to failure).

Analogous to: calibration_formulas.py in the metallicity project.
"""

import numpy as np
from gear_params import (
    GEAR, LOAD, MATERIAL, SN_CURVES,
    Kt_SOURCES, MEAN_STRESS_METHODS,
    DAMAGE_METHODS, SIZE_SURFACE_STANDARDS,
    LOAD_SPECTRUM, rho_F,
)


# ═══════════════════════════════════════════════════════════
# SWITCH 1: S-N Curve Source
# σa = σf' × (2Nf)^b  →  Nf = 0.5 × (σa/σf')^(1/b)
# ═══════════════════════════════════════════════════════════

def compute_fatigue_life_sn(stress_amplitude, sn_source="MatWeb"):
    """
    Returns Nf (cycles) for given stress amplitude and S-N curve source.
    If stress < endurance limit, returns ∞ (run-out).
    """
    curve = SN_CURVES[sn_source]
    sigma_f_prime = curve["sigma_f_prime"]
    b = curve["b"]
    fatigue_strength_1e6 = curve["fatigue_strength_at_1e6"]  # MPa, from SAE spreadsheet

    if stress_amplitude <= fatigue_strength_1e6:
        return np.inf  # run-out: stress below fatigue strength at 10^6 cycles

    # Basquin: σa = σf' × (2Nf)^b
    Nf = 0.5 * (stress_amplitude / sigma_f_prime) ** (1.0 / b)
    return Nf


# ═══════════════════════════════════════════════════════════
# SWITCH 2: Stress Concentration Factor Kf
# Kt → Kf via notch sensitivity q (Peterson or Neuber)
# ═══════════════════════════════════════════════════════════

def compute_Kt_geometric():
    """
    Theoretical stress concentration factor for gear root fillet.
    Engineering approximation calibrated to produce Kt in the range
    1.5--2.5 typical for standard involute gear tooth roots.
    NOTE: The original Dolan & Broghamer (1942) citation was incorrect;
    this is a geometric estimator, not a published formula. Design-grade
    Kt values require finite element analysis of the specific tooth geometry.
    Returns Kt (dimensionless).
    """
    mn = GEAR["module_mn"]
    r = rho_F  # root fillet radius
    h = 2.25 * mn  # full tooth depth
    t = np.pi * mn / 2  # tooth thickness at root (approx)

    # Kt ≈ 1 + 0.5 × sqrt(t/r) × (t/h)^0.3  (engineering estimate)
    Kt = 1.0 + 0.5 * np.sqrt(t / max(r, 0.1)) * (t / h) ** 0.3
    return Kt


def compute_Kf(method="Peterson"):
    """
    Fatigue notch factor Kf from Kt + notch sensitivity q.
    """
    Kt = compute_Kt_geometric()
    uts_MPa = MATERIAL["uts"]
    r = rho_F  # mm

    if method == "Peterson":
        # Peterson: a = 0.0254 × (2079/uts)^1.8  (mm), for steels
        # Kf = 1 + (Kt - 1) / (1 + a/r)
        a = 0.0254 * (2079.0 / uts_MPa) ** 1.8
        Kf = 1.0 + (Kt - 1.0) / (1.0 + a / max(r, 0.01))

    elif method == "Neuber":
        # Neuber: a' relates to material characteristic length
        # √a' ≈ 0.045 for steels with UTS ~1000 MPa (Peterson's Stress
        # Concentration Factors, 3rd ed., Table 1.1)
        a_prime = 0.002  # mm, calibrated for 42CrMo4 at UTS=1000 MPa
        Kf = 1.0 + (Kt - 1.0) / (1.0 + np.sqrt(a_prime) / max(r, 0.01))

    else:
        raise ValueError(f"Unknown Kf method: {method}")

    return Kf


# ═══════════════════════════════════════════════════════════
# NOMINAL STRESS (ISO 6336 Method B — bending)
# ═══════════════════════════════════════════════════════════

def compute_nominal_stress():
    """
    ISO 6336-3 Method B: root stress for spur gears.
    σF0 = (Ft / b·mn) × YFa × YSa × Yε × Yβ

    Returns σF0 (MPa) — nominal bending stress at tooth root.
    """
    mn = GEAR["module_mn"]
    b = GEAR["face_width"]
    Ft = LOAD["tangential_force"]
    z = GEAR["teeth_pinion"]
    alpha = np.radians(GEAR["pressure_angle"])

    # YFa — form factor, calibrated to ISO 6336-3 Figure 2/Table
    # For 20° PA, no profile shift, standard basic rack (ISO 53)
    # z=20→2.80, z=40→2.54, z=60→2.28
    YFa = 3.06 - 0.013 * z

    # YSa — stress correction factor (ISO 6336-3)
    # z=20→1.55, z=40→1.52, z=60→1.49
    YSa = 1.58 - 0.0015 * z

    # Yε — contact ratio factor
    # εα ≈ 1.7 for standard 20-tooth spur gear
    epsilon_alpha = 1.7
    Y_epsilon = 0.25 + 0.75 / epsilon_alpha  # Yε

    # Yβ — helix angle factor (= 1.0 for spur gears)
    Y_beta = 1.0

    sigma_F0 = (Ft / (b * mn)) * YFa * YSa * Y_epsilon * Y_beta
    return sigma_F0


# ═══════════════════════════════════════════════════════════
# SWITCH 3: Mean Stress Correction
# σeq = f(σa, σm, method)
# ═══════════════════════════════════════════════════════════

def apply_mean_stress_correction(sigma_max, stress_ratio_R, method="Goodman",
                                 sn_source="MatWeb"):
    """
    Convert σmax (with mean stress) to equivalent fully-reversed stress σar.
    σa = σmax × (1 - R) / 2   (stress amplitude)
    σm = σmax × (1 + R) / 2   (mean stress)

    sn_source: used for Morrow's σf' (fatigue strength coefficient).
    """
    sigma_a = sigma_max * (1.0 - stress_ratio_R) / 2.0
    sigma_m = sigma_max * (1.0 + stress_ratio_R) / 2.0

    uts = SN_CURVES[sn_source]["uts"]  # measured UTS of the selected S-N curve specimen
    sigma_f_prime = SN_CURVES[sn_source]["sigma_f_prime"]  # from selected S-N curve

    if method == "Goodman":
        # σa/σar + σm/uts = 1 → σar = σa / (1 - σm/uts)
        sigma_ar = sigma_a / max(1.0 - sigma_m / uts, 0.01)

    elif method == "Gerber":
        # σa/σar + (σm/uts)^2 = 1 → σar = σa / (1 - (σm/uts)^2)
        sigma_ar = sigma_a / max(1.0 - (sigma_m / uts) ** 2, 0.01)

    elif method == "Morrow":
        # σa/σar + σm/σf' = 1 → σar = σa / (1 - σm/σf_prime)
        sigma_ar = sigma_a / max(1.0 - sigma_m / sigma_f_prime, 0.01)

    elif method == "SWT":
        # SWT: σar = sqrt(σmax × σa)
        # (simplified strain-based; omits E for stress-only calculation)
        sigma_ar = np.sqrt(sigma_max * sigma_a)

    else:
        raise ValueError(f"Unknown mean stress method: {method}")

    return max(sigma_ar, 0.01)  # ensure positive


# ═══════════════════════════════════════════════════════════
# SWITCH 4: Cumulative Damage Rule
# ═══════════════════════════════════════════════════════════

def apply_damage_rule(Nf_constant, method="Miner_original"):
    """
    For constant amplitude loading, damage rule affects life via
    the failure threshold Dc (Miner sum at failure).
    Nf_effective = Nf_constant × Dc
    """
    if method == "Miner_original":
        Dc = 1.0
    elif method == "Miner_modified":
        Dc = 0.7  # AGMA/FKM recommend 0.7 for gearing
    else:
        raise ValueError(f"Unknown damage method: {method}")

    return Nf_constant * Dc


# ═══════════════════════════════════════════════════════════
# SWITCH 5: Size & Surface Factor
# ═══════════════════════════════════════════════════════════

def apply_size_surface_factor(sigma_ar, standard="ISO_6336", Rz_um=10.0, sn_source="MatWeb"):
    """
    Reduces endurance limit / allowable stress for size and surface effects.
    Returns modified stress (higher effective stress = shorter life).
    
    Rz_um: surface roughness in μm (SWITCH 6)
    """
    mn = GEAR["module_mn"]

    if standard == "ISO_6336":
        # YX: size factor (ISO 6336-3:2019, Table in §6.2)
        # For mn ≤ 10: YX = 1.00 (verified against standard table)
        if mn <= 10:
            YX = 1.0
        elif mn <= 30:
            YX = 0.95  # approximate from table
        else:
            YX = 0.90  # approximate from table

        # YRrelT: relative surface factor (ISO 6336-3:2019 §6.4, Annex B)
        # Valid range: 0.5 ≤ Rz ≤ 40 μm for quenched & tempered steels.
        # YRrelT = 1.674 − 0.529 × (Rz + 1)^0.1
        # For Rz > 40 μm: capped at the Rz=40 μm value (0.907);
        # the formula is not valid beyond this range.
        # Reference: Rz ≈ 10 μm → YRrelT ≈ 1.0 (standard test gear)
        if Rz_um < 1.0:
            YRrelT = 1.120  # plateau for polished surfaces (Rz < 1 μm)
        elif Rz_um <= 40.0:
            YRrelT = 1.674 - 0.529 * (Rz_um + 1.0) ** 0.1
        else:
            YRrelT = 1.674 - 0.529 * 41.0 ** 0.1  # cap at Rz=40 value ≈ 0.907
        
        sigma_effective = sigma_ar / max(YX * YRrelT, 0.5)

    elif standard == "AGMA_2001":
        # AGMA 2001-D04: size factor Ks (Clause 15, Figure 17)
        # For Pd ≥ 5 (module ≤ 5.08 mm): Ks = 1.0
        # Our gear: Pd = 25.4/3.0 = 8.47 → Ks = 1.0
        Ks = 1.0

        # Cf: surface condition factor (AGMA 2001-D04, Clause 16, Eq 16.1)
        # Cf = 1.0 for ground gears
        # Cf = 1 - 0.0058 × ln(Ra_uin) × ln(Sut_ksi) for all other finishes
        # Rf = surface roughness Ra, microinches (μin)
        # Sut = ultimate tensile strength, ksi
        #
        # Rz → Ra conversion: Ra ≈ Rz / 6 (typical for machined surfaces)
        # 1 μm = 39.37 μin
        #
        uts_ksi = SN_CURVES[sn_source]["uts"] / 6.895  # MPa → ksi
        if Rz_um <= 8:
            Cf = 1.0  # ground gears (AGMA 2001-D04 §16)
        else:
            Ra_uin = (Rz_um / 6.0) * 39.37  # Rz(μm) → Ra(μin)
            Cf = 1.0 - 0.0058 * np.log(Ra_uin) * np.log(uts_ksi)
            Cf = max(Cf, 0.7)  # lower bound for engineering validity

        # Kr: reliability factor for 99% survival
        Kr = 1.0
        sigma_effective = sigma_ar / max(Ks * Cf * Kr, 0.5)

    elif standard == "FKM":
        # FKM Guideline: Kd + KFσ
        # Kd: statistical size factor
        Kd = 1.0 - 0.05 * np.log10(max(mn * 10, 1))  # simplified
        # KFσ: surface roughness factor (FKM Guideline 7th ed.)
        # KFσ = 1 - a_R · log10(Rz) · log10(2·Rm / Rm,N,min)
        # a_R = 0.22 for steels; Rm,N,min = 400 MPa for wrought steels
        Rm_N_min = 400.0
        log_term = max(np.log10(2.0 * SN_CURVES[sn_source]["uts"] / Rm_N_min), 0.0)
        KF_sigma = max(1.0 - 0.22 * np.log10(max(Rz_um, 1.0)) * log_term, 0.55)
        sigma_effective = sigma_ar / max(Kd * KF_sigma, 0.5)

    else:
        raise ValueError(f"Unknown size/surface standard: {standard}")

    return sigma_effective


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE: Single combination → Nf
# ═══════════════════════════════════════════════════════════

def compute_fatigue_life(sn_source, kf_method, mean_stress_method,
                          damage_method, size_surface_standard,
                          rz_um=10.0, verbose=False):
    """
    Main pipeline: fixed gear + loading → through 6 switches → Nf (cycles).

    Returns:
        Nf: fatigue life in cycles (np.inf if run-out)
        diagnostics: dict with intermediate values for debugging
    """
    # 1. Nominal stress (ISO 6336-3 Method B, includes YSa stress concentration)
    sigma_nominal = compute_nominal_stress()
    # NOTE: Kf switch (v2.3) removed. YSa (≈1.55) already handles the tooth
    # root stress concentration in ISO 6336. Adding a separate Kf double-counts.
    # The notch sensitivity effect (YδrelT) operates on the strength side in
    # ISO 6336 and is not separately modeled here. See §2.5 and Limitations.

    # 2. Mean stress correction (SWITCH 2, was SWITCH 3)
    R = LOAD_SPECTRUM["stress_ratio_R"]
    sigma_ar = apply_mean_stress_correction(sigma_nominal, R, method=mean_stress_method,
                                            sn_source=sn_source)

    # 3. Size & surface factor (SWITCH 4 + SWITCH 5, were SWITCH 5 + 6)
    sigma_effective = apply_size_surface_factor(sigma_ar, standard=size_surface_standard, Rz_um=rz_um, sn_source=sn_source)

    # 4. S-N curve → Nf (SWITCH 1)
    Nf = compute_fatigue_life_sn(sigma_effective, sn_source=sn_source)

    # 5. Cumulative damage correction (SWITCH 3, was SWITCH 4)
    Nf = apply_damage_rule(Nf, method=damage_method)

    diagnostics = {
        "sigma_nominal": sigma_nominal,
        "sigma_ar": sigma_ar,
        "sigma_effective": sigma_effective,
        "Nf_raw": Nf,
    }

    if verbose:
        print(f"  σ_nominal={sigma_nominal:.1f} → Kf={Kf:.3f} → "
              f"σ_max={sigma_max:.1f} → σ_ar={sigma_ar:.1f} → "
              f"σ_eff={sigma_effective:.1f} → Nf={Nf:.1e}")

    return Nf, diagnostics


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test one combination
    Nf, diag = compute_fatigue_life(
        sn_source="MatWeb",
        kf_method="Peterson",
        mean_stress_method="Goodman",
        damage_method="Miner_original",
        size_surface_standard="ISO_6336",
        verbose=True,
    )
    print(f"\nFatigue life (MatWeb+Peterson+Goodman+Miner+ISO): {Nf:.1e} cycles")
    if Nf > 1e7:
        print("  → RUN-OUT (infinite life)")
    elif Nf < 1e4:
        print("  → LOW-CYCLE FATIGUE WARNING")
    else:
        print(f"  → HIGH-CYCLE FATIGUE, L10h = {Nf / (LOAD['speed_pinion'] * 60):.0f} hours")
