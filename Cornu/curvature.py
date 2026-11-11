
import numpy as np
from PIL import Image, ImageDraw
import scipy
import scipy.signal
import scipy.optimize
import scipy.odr
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from pathlib import Path
import csv
import lsfr

# height and width of the crop box
W = 826
H = 170

lam = 546e-6  # Wavelength in mm (Mercury green line)
g = 9.80665

# calibration scale factor (s = d / N) for [Cornu_1, Cornu_2, Cornu_3]
cal_known_distance_mm = [150, 150, 150]
cal_pixels_N = [3195.511, 3192.607, 3222.501]
cal_pixel_end_err = 6.16 / 2 # width of the line / 2

# m NOT M!
average_mass_kg = 0.50034
sigma_m_kg = 0.0003 # standard deviation of the masses

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
width_2a_mm = 51.82 # bar width
delta_2a_mm = 0.03 # bar width uncertainty
thickness_2b_mm = 6.50 # bar thickness
delta_2b_mm = 0.03 # bar thickness uncertainty

a_mm = width_2a_mm / 2.0
delta_a_mm = delta_2a_mm / 2.0
b_mm = thickness_2b_mm / 2.0
delta_b_mm = delta_2b_mm / 2.0


def linear_func(x, m, c):
    return m * x + c


def linear_odr(B, x):
    return B[0] * x + B[1]


folder = Path('C:/Users/profp/repos/experiments/Cornu/photos/Cornu_3')
set_idx = SET_INDEX.get(folder.name, 0)
d_cal_mm = cal_known_distance_mm[set_idx]
N_cal_px = cal_pixels_N[set_idx]
calibration_factor = d_cal_mm / N_cal_px  # mm/pixel
delta_s = (d_cal_mm / (N_cal_px ** 2)) * (np.sqrt(2) * cal_pixel_end_err)
L_mm = L_MM_BY_SET.get(folder.name, 100.0)

check_dir = folder.parent.parent / 'checks' / folder.name
plot_dir = folder.parent.parent / 'plots' / folder.name
intensity_dir = plot_dir / 'intensity'
lsfr_kappa_dir = plot_dir / 'lsfr_kappa'
lsfr_kappa_prime_dir = plot_dir / 'lsfr_kappa_prime'

for d in [check_dir, plot_dir, intensity_dir, lsfr_kappa_dir, lsfr_kappa_prime_dir]:
    d.mkdir(parents=True, exist_ok=True)

csv_path = plot_dir / 'curvatures.csv'

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file', 'kappa', 'kappa_prime', 'kappa_err', 'kappa_prime_err'])

masses_kg = []
kappa_values = []
kappa_errs = []
kappa_prime_values = []
kappa_prime_errs = []

for file in folder.glob('*'):

    if 'kg' not in file.name:
        continue

    n = float(file.stem.removesuffix('kg'))
    num_masses = n / 0.5
    M = num_masses * average_mass_kg

    original_image = Image.open(file)

    plt.figure(figsize=(10, 8))
    plt.title(f"1) Click rough center of the hyperbola for: {file.name}")
    plt.imshow(original_image)
    cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=0.8)
    
    # timeout=0 means it will wait indefinitely for click
    rough_clicks = plt.ginput(1, timeout=0)
    if not rough_clicks:
        plt.close()
        print(f"No click detected for {file.name}. Skipping.")
        continue

    rx, ry = rough_clicks[0]
    zoom_half = 250  # half-width of zoom window in pixels
    plt.xlim(rx - zoom_half, rx + zoom_half)
    plt.ylim(ry + zoom_half, ry - zoom_half)
    plt.title(f"2) Click exact center of the hyperbola for: {file.name}")
    plt.gcf().canvas.draw()

    clicks = plt.ginput(1, timeout=0) 
    plt.close()
    
    if not clicks:
        print(f"No click detected for {file.name}. Skipping.")
        continue

    center_x_click, center_y_click = clicks[0]
    X = int(center_x_click - (W / 2))
    Y = int(center_y_click - (H / 2))
    crop_box = (X, Y, X + W, Y + H)

    X_t = int(center_x_click - (H / 2))
    Y_t = int(center_y_click - (W / 2))
    crop_box_t = (X_t, Y_t, X_t + H, Y_t + W)

    # draw sanity check and crop
    
    img_check = original_image.copy()
    draw = ImageDraw.Draw(img_check)
    draw.rectangle(crop_box, outline='red', width=5)
    draw.rectangle(crop_box_t, outline='blue', width=5)

    check_path = check_dir / f'check_{file.name}'
    img_check.save(check_path)

    img_cropped = original_image.crop(crop_box).convert('L')
    img_array = np.array(img_cropped)

    img_cropped_t = original_image.crop(crop_box_t).convert('L')
    img_array_t = np.array(img_cropped_t)

    # get the intensity profile averaged across a 10 pixel band
    center_y = img_array.shape[0] // 2
    intensity_profile = np.mean(img_array[center_y - 5 : center_y + 5, :], axis=0)

    center_x_t = img_array_t.shape[1] // 2
    intensity_profile_t = np.mean(img_array_t[:, center_x_t - 5 : center_x_t + 5], axis=1)

    # find dark fringes (peaks of the inverse profile)
    inverted_profile = -intensity_profile
    peaks, properties = scipy.signal.find_peaks(inverted_profile, distance=15, prominence=5)

    inverted_profile_t = -intensity_profile_t
    peaks_t, properties_t = scipy.signal.find_peaks(inverted_profile_t, distance=15, prominence=5)

    # get centre coordinate
    center_x = img_array.shape[1] / 2
    center_y_t = img_array_t.shape[0] / 2
    center_margin = 25

    peaks = peaks[np.abs(peaks - center_x) > center_margin]
    peaks_t = peaks_t[np.abs(peaks_t - center_y_t) > center_margin]

    # find peaks on each of the 10 rows individually for longitudinal fringes
    row_left_peaks = []
    row_right_peaks = []
    for row in img_array[center_y - 5 : center_y + 5, :]:
        r_peaks, _ = scipy.signal.find_peaks(-row.astype(float), distance=15, prominence=5)
        r_peaks = r_peaks[np.abs(r_peaks - center_x) > center_margin]
        row_left_peaks.append(r_peaks[r_peaks < center_x][::-1])
        row_right_peaks.append(r_peaks[r_peaks > center_x])

    # calculate 2x_n, x_n^2, and uncertainty in x_n^2
    x_n_values = []
    x_n_uncertainties = []
    x_n_squared_values = []
    x_n_sq_uncertainties = []
    n_values = []

    min_length = min(min(len(l), len(r)) for l, r in zip(row_left_peaks, row_right_peaks))
    for n in range(min_length):
        row_distances_2xn = np.array([row_right_peaks[i][n] - row_left_peaks[i][n] for i in range(10)])
        pixel_distance_2xn = np.mean(row_distances_2xn)
        physical_distance_2xn = pixel_distance_2xn * calibration_factor

        x_n = physical_distance_2xn / 2.0
        x_n_squared = x_n ** 2

        sigma_2xn_stat = np.std(row_distances_2xn, ddof=1)
        sigma_2xn_quant = np.sqrt(2.0) / np.sqrt(12.0)  # quantization error from both ends
        sigma_2xn_p = np.sqrt(sigma_2xn_stat ** 2 + sigma_2xn_quant ** 2)

        sigma_xn_p = sigma_2xn_p / 2.0
        sigma_xn = calibration_factor * sigma_xn_p
        sigma_xn_sq = 2.0 * x_n * sigma_xn

        # assuming the first clear dark fringe pair is n=1
        fringe_order = n + 1

        x_n_values.append(x_n)
        x_n_uncertainties.append(sigma_xn)
        x_n_squared_values.append(x_n_squared)
        x_n_sq_uncertainties.append(sigma_xn_sq)
        n_values.append(fringe_order)

    # find peaks on each of the 10 columns individually for transverse fringes
    col_top_peaks = []
    col_bottom_peaks = []
    for col in img_array_t[:, center_x_t - 5 : center_x_t + 5].T:
        c_peaks, _ = scipy.signal.find_peaks(-col.astype(float), distance=15, prominence=5)
        c_peaks = c_peaks[np.abs(c_peaks - center_y_t) > center_margin]
        col_top_peaks.append(c_peaks[c_peaks < center_y_t][::-1])
        col_bottom_peaks.append(c_peaks[c_peaks > center_y_t])

    # calculate 2y_n, y_n^2, and uncertainty in y_n^2
    y_n_values = []
    y_n_uncertainties = []
    y_n_squared_values = []
    y_n_sq_uncertainties = []
    n_values_t = []

    min_length_t = min(min(len(t), len(b)) for t, b in zip(col_top_peaks, col_bottom_peaks))
    for n in range(min_length_t):
        col_distances_2yn = np.array([col_bottom_peaks[i][n] - col_top_peaks[i][n] for i in range(10)])
        pixel_distance_2yn = np.mean(col_distances_2yn)
        physical_distance_2yn = pixel_distance_2yn * calibration_factor

        y_n = physical_distance_2yn / 2.0
        y_n_squared = y_n ** 2

        sigma_2yn_stat = np.std(col_distances_2yn, ddof=1)
        sigma_2yn_quant = np.sqrt(2.0) / np.sqrt(12.0)  # quantization error from both ends
        sigma_2yn_p = np.sqrt(sigma_2yn_stat ** 2 + sigma_2yn_quant ** 2)

        sigma_yn_p = sigma_2yn_p / 2.0
        sigma_yn = calibration_factor * sigma_yn_p
        sigma_yn_sq = 2.0 * y_n * sigma_yn

        fringe_order = n + 1

        y_n_values.append(y_n)
        y_n_uncertainties.append(sigma_yn)
        y_n_squared_values.append(y_n_squared)
        y_n_sq_uncertainties.append(sigma_yn_sq)
        n_values_t.append(fringe_order)

    # Graph 1 - least squares fit(x_n^2 = m1 * n + c1)
    n_values_arr = np.asarray(n_values, dtype=float)
    x_n_sq_arr = np.asarray(x_n_squared_values, dtype=float)
    x_n_sq_err_arr = np.asarray(x_n_sq_uncertainties, dtype=float)

    popt, pcov = scipy.optimize.curve_fit(
        linear_func,
        n_values_arr,
        x_n_sq_arr,
        sigma=x_n_sq_err_arr,
        absolute_sigma=True
    )
    m1, c1 = popt
    sigma_m1, sigma_c1 = np.sqrt(np.diag(pcov))
    kappa = lam / m1
    kappa_err = kappa * np.abs(sigma_m1 / m1)

    ss_res = np.sum((x_n_sq_arr - linear_func(n_values_arr, m1, c1)) ** 2)
    ss_tot = np.sum((x_n_sq_arr - np.mean(x_n_sq_arr)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Graph 2 - least squares fit for transverse fringes (y_n^2 = m1_t * n + c1_t)
    n_values_t_arr = np.asarray(n_values_t, dtype=float)
    y_n_sq_arr = np.asarray(y_n_squared_values, dtype=float)
    y_n_sq_err_arr = np.asarray(y_n_sq_uncertainties, dtype=float)

    popt_t, pcov_t = scipy.optimize.curve_fit(
        linear_func,
        n_values_t_arr,
        y_n_sq_arr,
        sigma=y_n_sq_err_arr,
        absolute_sigma=True
    )
    m1_t, c1_t = popt_t
    sigma_m1_t, sigma_c1_t = np.sqrt(np.diag(pcov_t))
    kappa_prime = lam / m1_t
    kappa_prime_err = kappa_prime * np.abs(sigma_m1_t / m1_t)

    ss_res_t = np.sum((y_n_sq_arr - linear_func(n_values_t_arr, m1_t, c1_t)) ** 2)
    ss_tot_t = np.sum((y_n_sq_arr - np.mean(y_n_sq_arr)) ** 2)
    r_squared_t = 1.0 - (ss_res_t / ss_tot_t) if ss_tot_t != 0 else 0.0

    # Output for lab book
    print(f"\n--- {folder.name} | Mass M = {M:.5f} kg ({file.name}) ---")
    print(f"kappa       = {kappa:.6e} ± {kappa_err:.6e} mm^-1  ({kappa * 1e3:.4f} ± {kappa_err * 1e3:.4f} m^-1)")
    print(f"kappa_prime = {kappa_prime:.6e} ± {kappa_prime_err:.6e} mm^-1  ({kappa_prime * 1e3:.4f} ± {kappa_prime_err * 1e3:.4f} m^-1)")
    print(f"{'n':>3} | {'x_n (mm)':>10} | {'sigma_x_n (mm)':>14} | {'x_n^2 (mm^2)':>12} | {'sigma_x_n^2 (mm^2)':>18}")
    print("-" * 68)
    for fn, xn_val, sxn_val, xnsq_val, sxnsq_val in zip(
        n_values, x_n_values, x_n_uncertainties, x_n_squared_values, x_n_sq_uncertainties
    ):
        print(f"{fn:>3d} | {xn_val:>10.4f} | {sxn_val:>14.4f} | {xnsq_val:>12.4f} | {sxnsq_val:>18.4f}")

    masses_kg.append(M)
    kappa_values.append(kappa)
    kappa_errs.append(kappa_err)
    kappa_prime_values.append(kappa_prime)
    kappa_prime_errs.append(kappa_prime_err)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file.stem, kappa, kappa_prime, kappa_err, kappa_prime_err])

    # generate LSFR plots for x_n^2 and y_n^2 against n
    lsfr_path = lsfr_kappa_dir / f'{file.stem}.png'
    lsfr.create_plot(
        n_values_arr,
        x_n_sq_arr,
        [m1, c1],
        [sigma_m1, sigma_c1],
        y_uncertainties=x_n_sq_err_arr,
        r_squared=r_squared,
        plot_title=f'Longitudinal $x_n^2$ vs Fringe Order $n$ ({file.stem})',
        x_label=r'Fringe Order $n$',
        y_label=r'$x_n^2$ / $\mathrm{mm}^2$',
        figure_name=str(lsfr_path)
    )

    lsfr_path_t = lsfr_kappa_prime_dir / f'{file.stem}.png'
    lsfr.create_plot(
        n_values_t_arr,
        y_n_sq_arr,
        [m1_t, c1_t],
        [sigma_m1_t, sigma_c1_t],
        y_uncertainties=y_n_sq_err_arr,
        r_squared=r_squared_t,
        plot_title=f'Transverse $y_n^2$ vs Fringe Order $n$ ({file.stem})',
        x_label=r'Fringe Order $n$',
        y_label=r'$y_n^2$ / $\mathrm{mm}^2$',
        figure_name=str(lsfr_path_t)
    )

    # create plot for each to verify
    plt.figure()
    plt.plot(intensity_profile, color="tab:blue", label="Longitudinal")
    plt.plot(peaks, intensity_profile[peaks], "x", color="red")
    plt.plot(intensity_profile_t, color="tab:orange", label="Transverse")
    plt.plot(peaks_t, intensity_profile_t[peaks_t], "x", color="green")
    plt.title("Intensity Profile with Detected Dark Fringes")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Intensity")
    plt.legend(loc="upper left")

    # display calculated values in top-right corner
    plt.text(
        0.98, 0.98,
        f"$\\kappa = {kappa:.6f}\\ \\mathrm{{mm}}^{{-1}}$ ($R^2 = {r_squared:.4f}$)\n"
        f"$\\kappa' = {kappa_prime:.6f}\\ \\mathrm{{mm}}^{{-1}}$ ($R^2 = {r_squared_t:.4f}$)",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plot_path = intensity_dir / f'{file.stem}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

if len(kappa_values) >= 2:
    # Sort arrays by mass M
    sort_idx = np.argsort(masses_kg)
    masses_arr = np.asarray(masses_kg, dtype=float)[sort_idx]
    kappa_arr = np.asarray(kappa_values, dtype=float)[sort_idx]
    kappa_err_arr = np.asarray(kappa_errs, dtype=float)[sort_idx]
    kappa_prime_arr = np.asarray(kappa_prime_values, dtype=float)[sort_idx]
    kappa_prime_err_arr = np.asarray(kappa_prime_errs, dtype=float)[sort_idx]

    # Convert lengths and curvatures to SI units (m, m^-1)
    L_m = L_mm * 1e-3
    delta_L_m = delta_L_mm * 1e-3
    a_m = a_mm * 1e-3
    delta_a_m = delta_a_mm * 1e-3
    b_m = b_mm * 1e-3
    delta_b_m = delta_b_mm * 1e-3

    kappa_vals_si = kappa_arr * 1e3
    sigma_kappa_vals_si = kappa_err_arr * 1e3
    kappa_prime_vals_si = kappa_prime_arr * 1e3
    sigma_kappa_prime_vals_si = kappa_prime_err_arr * 1e3


    # Graph 3 — ODR Fit of kappa vs G (Finding E and kappa_0)
    N_m = masses_arr / average_mass_kg
    sigma_M_vals = np.sqrt(N_m) * sigma_m_kg
    G_vals = masses_arr * g * L_m
    sigma_G_vals = sigma_M_vals * g * L_m

    odr_model = scipy.odr.Model(linear_odr)
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
        plot_title=f"Longitudinal Curvature $\\kappa$ vs Bending Moment $G$ ({folder.name})",
        x_label=r"Bending Moment $G$ / $\mathrm{N\cdot m}$",
        y_label=r"$\kappa$ / $\mathrm{m}^{-1}$",
        figure_name=str(plot_dir / "youngs_modulus_odr.png")
    )


    # Graph 4 — ODR Fit of kappa' vs kappa (Finding sigma and kappa_0')
    odr_data_sigma = scipy.odr.RealData(
        kappa_vals_si, kappa_prime_vals_si,
        sx=sigma_kappa_vals_si, sy=sigma_kappa_prime_vals_si
    )
    odr_obj_sigma = scipy.odr.ODR(odr_data_sigma, odr_model, beta0=[0.3, 0.0])
    out_sigma = odr_obj_sigma.run()
    poisson_sigma, c_int = out_sigma.beta
    sigma_poisson, sigma_c_int = out_sigma.sd_beta

    kappa_0_prime = c_int + poisson_sigma * kappa_0
    sigma_prod = np.abs(poisson_sigma * kappa_0) * np.sqrt(
        (sigma_poisson / poisson_sigma) ** 2 + (sigma_kappa_0 / kappa_0) ** 2
    )
    sigma_kappa_0_prime = np.sqrt(sigma_c_int ** 2 + sigma_prod ** 2)

    lsfr.create_plot(
        x_data=kappa_vals_si,
        y_data=kappa_prime_vals_si,
        parameters=[poisson_sigma, c_int],
        parameter_uncertainties=[sigma_poisson, sigma_c_int],
        x_uncertainties=sigma_kappa_vals_si,
        y_uncertainties=sigma_kappa_prime_vals_si,
        plot_title=f"Transverse vs Longitudinal Curvature ({folder.name})",
        x_label=r"$\kappa$ / $\mathrm{m}^{-1}$",
        y_label=r"$\kappa'$ / $\mathrm{m}^{-1}$",
        figure_name=str(plot_dir / "poisson_ratio_odr.png")
    )

    # Systematic Uncertainties
    delta_I = np.sqrt(
        ((4.0 / 3.0) * (b_m ** 3) * delta_a_m) ** 2
        + (4.0 * a_m * (b_m ** 2) * delta_b_m) ** 2
    )

    delta_E = E * np.sqrt(
        (delta_L_m / L_m) ** 2
        + (delta_I / I_m4) ** 2
        + (2.0 * (delta_s / calibration_factor)) ** 2
    )

    delta_kappa_0 = np.abs(kappa_0) * (2.0 * (delta_s / calibration_factor))
    delta_kappa_0_prime = np.abs(kappa_0_prime) * (2.0 * (delta_s / calibration_factor))
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

    summary_csv_path = plot_dir / "final_results_summary.csv"
    with open(summary_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['parameter', 'value', 'stat_err', 'syst_err', 'total_err'])
        for row in summary_rows:
            writer.writerow(row)

