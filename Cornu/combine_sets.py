import csv
from pathlib import Path
import numpy as np
import scipy.odr
import lsfr

g = 9.80665

# calibration scale factor (s = d / N) for [Cornu_1, Cornu_2, Cornu_3]
cal_known_distance_mm = [150, 150, 150]
cal_pixels_N = [3195.511, 3192.607, 3222.501]
cal_pixel_end_err = 6.16 / 2  # width of the line / 2

# m NOT M!
average_mass_kg = 0.50034
sigma_m_kg = 0.0003  # standard deviation of the masses

SET_INDEX = {
    'Cornu_1': 0,
    'Cornu_2': 1,
    'Cornu_3': 2,
}

L_MM_BY_SET = {
    'Cornu_1': 158.0,  # 15.8 cm
    'Cornu_2': 100.0,  # 10.0 cm
    'Cornu_3': 180.0,  # 18.0 cm (not entirely sure)
}
delta_L_mm = np.sqrt(2) * 0.5

# bar dimensions
width_2a_mm = 51.82   # bar width
delta_2a_mm = 0.03    # bar width uncertainty
thickness_2b_mm = 6.50  # bar thickness
delta_2b_mm = 0.03    # bar thickness uncertainty

a_mm = width_2a_mm / 2.0
delta_a_mm = delta_2a_mm / 2.0
b_mm = thickness_2b_mm / 2.0
delta_b_mm = delta_2b_mm / 2.0

a_m = a_mm * 1e-3
delta_a_m = delta_a_mm * 1e-3
b_m = b_mm * 1e-3
delta_b_m = delta_b_mm * 1e-3
delta_L_m = delta_L_mm * 1e-3


def linear_odr(B, x):
    return B[0] * x + B[1]


base_plots_dir = Path('C:/Users/profp/repos/experiments/Cornu/plots')
combined_dir = base_plots_dir / 'combined'
combined_dir.mkdir(parents=True, exist_ok=True)

G_list = []
sigma_G_list = []
kappa_si_list = []
sigma_kappa_si_list = []
kappa_prime_si_list = []
sigma_kappa_prime_si_list = []

loaded_sets = []
L_m_per_set = []
cal_factor_per_set = []
delta_s_per_set = []
combined_rows = []
kappa_0_per_set = []
c_int_per_set = []
odr_model = scipy.odr.Model(linear_odr)

for set_name in ['Cornu_1', 'Cornu_2', 'Cornu_3']:
    csv_path = base_plots_dir / set_name / 'curvatures.csv'
    if not csv_path.exists():
        print(f"Skipping {set_name}: {csv_path} not found.")
        continue

    set_idx = SET_INDEX[set_name]
    d_cal_mm = cal_known_distance_mm[set_idx]
    N_cal_px = cal_pixels_N[set_idx]
    cal_factor = d_cal_mm / N_cal_px
    delta_s = (d_cal_mm / (N_cal_px ** 2)) * (np.sqrt(2) * cal_pixel_end_err)
    L_m = L_MM_BY_SET[set_name] * 1e-3

    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        print(f"Skipping {set_name}: {csv_path} is empty.")
        continue

    loaded_sets.append(set_name)
    L_m_per_set.append(L_m)
    cal_factor_per_set.append(cal_factor)
    delta_s_per_set.append(delta_s)

    set_G = []
    set_sigma_G = []
    set_kappa = []
    set_sigma_kappa = []
    set_kappa_prime = []
    set_sigma_kappa_prime = []
    set_file_stems = []
    set_masses = []

    for row in rows:
        file_stem = row['file']
        n = float(file_stem.removesuffix('kg'))
        num_masses = n / 0.5
        M = num_masses * average_mass_kg
        sigma_M = np.sqrt(num_masses) * sigma_m_kg

        G = M * g * L_m
        sigma_G = sigma_M * g * L_m

        # Convert curvatures from mm^-1 to m^-1
        kappa_si = float(row['kappa']) * 1e3
        sigma_kappa_si = float(row['kappa_err']) * 1e3
        kappa_prime_si = float(row['kappa_prime']) * 1e3
        sigma_kappa_prime_si = float(row['kappa_prime_err']) * 1e3

        set_G.append(G)
        set_sigma_G.append(sigma_G)
        set_kappa.append(kappa_si)
        set_sigma_kappa.append(sigma_kappa_si)
        set_kappa_prime.append(kappa_prime_si)
        set_sigma_kappa_prime.append(sigma_kappa_prime_si)
        set_file_stems.append(file_stem)
        set_masses.append(M)

    # Estimate per-set residual curvature intercept (since moving hangers to change L alters G_0)
    set_G_arr = np.asarray(set_G, dtype=float)
    set_sG_arr = np.asarray(set_sigma_G, dtype=float)
    set_k_arr = np.asarray(set_kappa, dtype=float)
    set_sk_arr = np.asarray(set_sigma_kappa, dtype=float)
    set_kp_arr = np.asarray(set_kappa_prime, dtype=float)
    set_skp_arr = np.asarray(set_sigma_kappa_prime, dtype=float)

    out_set_E = scipy.odr.ODR(
        scipy.odr.RealData(
            set_G_arr, set_k_arr,
            sx=np.where(set_sG_arr == 0, 1e-12, set_sG_arr),
            sy=set_sk_arr
        ),
        odr_model,
        beta0=[1.0, 0.0]
    ).run()
    k0_set = out_set_E.beta[1]
    kappa_0_per_set.append(k0_set)

    out_set_p = scipy.odr.ODR(
        scipy.odr.RealData(set_k_arr, set_kp_arr, sx=set_sk_arr, sy=set_skp_arr),
        odr_model,
        beta0=[0.3, 0.0]
    ).run()
    c_int_set = out_set_p.beta[1]
    c_int_per_set.append(c_int_set)

    for i in range(len(set_G)):
        G_list.append(set_G[i])
        sigma_G_list.append(set_sigma_G[i])
        # Store residual-subtracted curvatures first; mean intercept is added back below
        kappa_si_list.append(set_kappa[i] - k0_set)
        sigma_kappa_si_list.append(set_sigma_kappa[i])
        kappa_prime_si_list.append(set_kappa_prime[i] - c_int_set)
        sigma_kappa_prime_si_list.append(set_sigma_kappa_prime[i])

        combined_rows.append([
            set_name, set_file_stems[i], set_masses[i], set_G[i], set_sigma_G[i],
            set_kappa[i], set_sigma_kappa[i], set_kappa_prime[i], set_sigma_kappa_prime[i]
        ])

if len(G_list) >= 2:
    print(f"Combining {len(G_list)} data points from sets: {', '.join(loaded_sets)}")

    mean_k0 = float(np.mean(kappa_0_per_set))
    mean_c_int = float(np.mean(c_int_per_set))

    # Sort all points by Bending Moment G
    sort_idx = np.argsort(G_list)
    G_vals = np.asarray(G_list, dtype=float)[sort_idx]
    sigma_G_vals = np.asarray(sigma_G_list, dtype=float)[sort_idx]
    kappa_vals_si = np.asarray(kappa_si_list, dtype=float)[sort_idx] + mean_k0
    sigma_kappa_vals_si = np.asarray(sigma_kappa_si_list, dtype=float)[sort_idx]
    kappa_prime_vals_si = np.asarray(kappa_prime_si_list, dtype=float)[sort_idx] + mean_c_int
    sigma_kappa_prime_vals_si = np.asarray(sigma_kappa_prime_si_list, dtype=float)[sort_idx]

    # Save combined data table
    combined_csv_path = combined_dir / 'combined_curvatures.csv'
    with open(combined_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'set', 'file', 'mass_kg', 'G_Nm', 'sigma_G_Nm',
            'kappa_m_inv', 'sigma_kappa_m_inv', 'kappa_prime_m_inv', 'sigma_kappa_prime_m_inv'
        ])
        for idx in sort_idx:
            writer.writerow(combined_rows[idx])

    # Graph 3 — Combined ODR Fit of kappa vs G (Finding E and kappa_0)
    odr_data_E = scipy.odr.RealData(
        G_vals, kappa_vals_si,
        sx=np.where(sigma_G_vals == 0, 1e-12, sigma_G_vals),
        sy=sigma_kappa_vals_si
    )
    odr_obj_E = scipy.odr.ODR(odr_data_E, odr_model, beta0=[1.0, 0.0])
    out_E = odr_obj_E.run()
    m_G, kappa_0 = out_E.beta
    sigma_m_G, sigma_kappa_0 = out_E.sd_beta

    I_m4 = (4.0 / 3.0) * a_m * (b_m ** 3)
    E = 1.0 / (m_G * I_m4)
    sigma_E = E * np.abs(sigma_m_G / m_G)

    lsfr.create_plot(
        x_data=G_vals,
        y_data=kappa_vals_si,
        parameters=[m_G, kappa_0],
        parameter_uncertainties=[sigma_m_G, sigma_kappa_0],
        x_uncertainties=sigma_G_vals,
        y_uncertainties=sigma_kappa_vals_si,
        plot_title=f"Longitudinal Curvature $\\kappa$ vs Bending Moment $G$ (Combined: {', '.join(loaded_sets)})",
        x_label=r"Bending Moment $G$ / $\mathrm{N\cdot m}$",
        y_label=r"$\kappa$ / $\mathrm{m}^{-1}$",
        figure_name=str(combined_dir / "youngs_modulus_odr.png")
    )

    # Graph 4 — Combined ODR Fit of kappa' vs kappa (Finding sigma and kappa_0')
    odr_data_sigma = scipy.odr.RealData(
        kappa_vals_si, kappa_prime_vals_si,
        sx=sigma_kappa_vals_si, sy=sigma_kappa_prime_vals_si
    )
    odr_obj_sigma = scipy.odr.ODR(odr_data_sigma, odr_model, beta0=[0.3, 0.0])
    out_sigma = odr_obj_sigma.run()
    poisson_sigma, c_int = out_sigma.beta
    sigma_poisson, sigma_c_int = out_sigma.sd_beta

    kappa_0_prime = c_int + poisson_sigma * kappa_0
    cov_sigma_c = out_sigma.cov_beta[0, 1] * out_sigma.res_var
    sigma_kappa_0_prime = np.sqrt(
        max(
            0.0,
            sigma_c_int ** 2
            + (kappa_0 * sigma_poisson) ** 2
            + (poisson_sigma * sigma_kappa_0) ** 2
            + 2.0 * kappa_0 * cov_sigma_c,
        )
    )

    lsfr.create_plot(
        x_data=kappa_vals_si,
        y_data=kappa_prime_vals_si,
        parameters=[poisson_sigma, c_int],
        parameter_uncertainties=[sigma_poisson, sigma_c_int],
        x_uncertainties=sigma_kappa_vals_si,
        y_uncertainties=sigma_kappa_prime_vals_si,
        plot_title=f"Transverse vs Longitudinal Curvature (Combined: {', '.join(loaded_sets)})",
        x_label=r"$\kappa$ / $\mathrm{m}^{-1}$",
        y_label=r"$\kappa'$ / $\mathrm{m}^{-1}$",
        figure_name=str(combined_dir / "poisson_ratio_odr.png")
    )

    # Systematic Uncertainties (using mean L, s, delta_s across the combined sets)
    mean_L_m = float(np.mean(L_m_per_set))
    mean_cal_factor = float(np.mean(cal_factor_per_set))
    mean_delta_s = float(np.mean(delta_s_per_set))

    delta_I = np.sqrt(
        ((4.0 / 3.0) * (b_m ** 3) * delta_a_m) ** 2
        + (4.0 * a_m * (b_m ** 2) * delta_b_m) ** 2
    )

    delta_E = E * np.sqrt(
        (delta_L_m / mean_L_m) ** 2
        + (delta_I / I_m4) ** 2
        + (2.0 * (mean_delta_s / mean_cal_factor)) ** 2
    )

    delta_kappa_0 = np.abs(kappa_0) * (2.0 * (mean_delta_s / mean_cal_factor))
    delta_kappa_0_prime = np.abs(kappa_0_prime) * (2.0 * (mean_delta_s / mean_cal_factor))
    delta_poisson = 0.0

    total_E = np.sqrt(sigma_E ** 2 + delta_E ** 2)
    total_poisson = np.sqrt(sigma_poisson ** 2 + delta_poisson ** 2)
    total_kappa_0 = np.sqrt(sigma_kappa_0 ** 2 + delta_kappa_0 ** 2)
    total_kappa_0_prime = np.sqrt(sigma_kappa_0_prime ** 2 + delta_kappa_0_prime ** 2)

    summary_rows = [
        ("Young's Modulus E (Pa)", E, sigma_E, delta_E, total_E),
        ("Poisson's Ratio sigma", poisson_sigma, sigma_poisson, delta_poisson, total_poisson),
        ("Longitudinal Residual Curvature kappa_0 (m^-1)", kappa_0, sigma_kappa_0, delta_kappa_0, total_kappa_0),
        ("Transverse Residual Curvature kappa_0' (m^-1)", kappa_0_prime, sigma_kappa_0_prime, delta_kappa_0_prime, total_kappa_0_prime),
    ]

    print("\n" + "=" * 95)
    print(f"{'Parameter':<48} | {'Value':>12} | {'Stat Err':>10} | {'Syst Err':>10} | {'Total Err':>10}")
    print("-" * 95)
    for name, val, stat_e, syst_e, tot_e in summary_rows:
        print(f"{name:<48} | {val:>12.4e} | {stat_e:>10.4e} | {syst_e:>10.4e} | {tot_e:>10.4e}")
    print("=" * 95 + "\n")

    summary_csv_path = combined_dir / "final_results_summary.csv"
    with open(summary_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['parameter', 'value', 'stat_err', 'syst_err', 'total_err'])
        for row in summary_rows:
            writer.writerow(row)
else:
    print("Not enough data points found across sets to perform combined ODR fits.")
