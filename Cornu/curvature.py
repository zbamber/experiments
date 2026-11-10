
import numpy as np
from PIL import Image, ImageDraw
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import lsfr

W = 826
H = 170

calibration_factor = 0.0469  # mm per pixel

folder = Path('C:/Users/profp/repos/experiments/Cornu/photos/Cornu_1')

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

kappa_values = []
kappa_prime_values = []
kappa_prime_errs = []

for file in folder.glob('*'):

    if 'kg' not in file.name:
        continue

    original_image = Image.open(file)

    plt.figure(figsize=(10, 8))
    plt.title(f"Click the center of the hyperbola for: {file.name}")
    plt.imshow(original_image)
    
    # timeout=0 means it will wait indefinitely for click
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

    # calculate 2x_n, x_n^2, and uncertainty in x_n
    x_n_squared_values = []
    x_n_uncertainties = []
    n_values = []

    min_length = min(min(len(l), len(r)) for l, r in zip(row_left_peaks, row_right_peaks))
    for n in range(min_length):
        row_distances_2xn = np.array([row_right_peaks[i][n] - row_left_peaks[i][n] for i in range(10)])
        pixel_distance_2xn = np.mean(row_distances_2xn)
        physical_distance_2xn = pixel_distance_2xn * calibration_factor

        x_n = physical_distance_2xn / 2.0
        x_n_squared = x_n ** 2

        row_xn_pixels = row_distances_2xn / 2.0
        dx_n_stat = np.std(row_xn_pixels, ddof=1) / np.sqrt(10)
        dx_n_quant = 0.5
        dx_n_pixels = np.sqrt(dx_n_stat ** 2 + dx_n_quant ** 2)
        x_n_err = dx_n_pixels * calibration_factor

        # assuming the first clear dark fringe pair is n=1
        fringe_order = n + 1

        x_n_squared_values.append(x_n_squared)
        x_n_uncertainties.append(x_n_err)
        n_values.append(fringe_order)

    # find peaks on each of the 10 columns individually for transverse fringes
    col_top_peaks = []
    col_bottom_peaks = []
    for col in img_array_t[:, center_x_t - 5 : center_x_t + 5].T:
        c_peaks, _ = scipy.signal.find_peaks(-col.astype(float), distance=15, prominence=5)
        c_peaks = c_peaks[np.abs(c_peaks - center_y_t) > center_margin]
        col_top_peaks.append(c_peaks[c_peaks < center_y_t][::-1])
        col_bottom_peaks.append(c_peaks[c_peaks > center_y_t])

    # calculate 2y_n, y_n^2, and uncertainty in y_n
    y_n_squared_values = []
    y_n_uncertainties = []
    n_values_t = []

    min_length_t = min(min(len(t), len(b)) for t, b in zip(col_top_peaks, col_bottom_peaks))
    for n in range(min_length_t):
        col_distances_2yn = np.array([col_bottom_peaks[i][n] - col_top_peaks[i][n] for i in range(10)])
        pixel_distance_2yn = np.mean(col_distances_2yn)
        physical_distance_2yn = pixel_distance_2yn * calibration_factor

        y_n = physical_distance_2yn / 2.0
        y_n_squared = y_n ** 2

        col_yn_pixels = col_distances_2yn / 2.0
        dy_n_stat = np.std(col_yn_pixels, ddof=1) / np.sqrt(10)
        dy_n_quant = 0.5
        dy_n_pixels = np.sqrt(dy_n_stat ** 2 + dy_n_quant ** 2)
        y_n_err = dy_n_pixels * calibration_factor

        fringe_order = n + 1

        y_n_squared_values.append(y_n_squared)
        y_n_uncertainties.append(y_n_err)
        n_values_t.append(fringe_order)

    # find curvature
    lam = 546e-6  # wavelength in mm (Mercury)

    res = scipy.stats.linregress(x_n_squared_values, n_values)
    slope, intercept, r_value, p_value, std_err = res
    kappa = slope * lam
    kappa_err = std_err * lam
    r_squared = r_value ** 2

    res_t = scipy.stats.linregress(y_n_squared_values, n_values_t)
    slope_t, intercept_t, r_value_t, p_value_t, std_err_t = res_t
    kappa_prime = slope_t * lam
    kappa_prime_err = std_err_t * lam
    r_squared_t = r_value_t ** 2

    kappa_values.append(kappa)
    kappa_prime_values.append(kappa_prime)
    kappa_prime_errs.append(kappa_prime_err)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([file.stem, kappa, kappa_prime, kappa_err, kappa_prime_err])

    # generate LSFR plots for n against x_n^2 and y_n^2
    lsfr_path = lsfr_kappa_dir / f'{file.stem}.png'
    lsfr.create_plot(
        x_n_squared_values,
        n_values,
        [slope, intercept],
        [std_err, res.intercept_stderr],
        r_squared=r_squared,
        plot_title=f'Longitudinal Fringe Order $n$ vs $x_n^2$ ({file.stem})',
        x_label=r'$x_n^2$ / $\mathrm{mm}^2$',
        y_label=r'Fringe Order $n$',
        figure_name=str(lsfr_path)
    )

    lsfr_path_t = lsfr_kappa_prime_dir / f'{file.stem}.png'
    lsfr.create_plot(
        y_n_squared_values,
        n_values_t,
        [slope_t, intercept_t],
        [std_err_t, res_t.intercept_stderr],
        r_squared=r_squared_t,
        plot_title=f'Transverse Fringe Order $n$ vs $y_n^2$ ({file.stem})',
        x_label=r'$y_n^2$ / $\mathrm{mm}^2$',
        y_label=r'Fringe Order $n$',
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
    res_sigma = scipy.stats.linregress(kappa_values, kappa_prime_values)
    lsfr.create_plot(
        kappa_values,
        kappa_prime_values,
        [res_sigma.slope, res_sigma.intercept],
        [res_sigma.stderr, res_sigma.intercept_stderr],
        y_uncertainties=kappa_prime_errs,
        r_squared=res_sigma.rvalue ** 2,
        plot_title=f"Transverse vs Longitudinal Curvature ({folder.name})",
        x_label=r"$\kappa$ / $\mathrm{mm}^{-1}$",
        y_label=r"$\kappa'$ / $\mathrm{mm}^{-1}$",
        figure_name=str(plot_dir / "poisson_ratio_lsfr.png")
    )
