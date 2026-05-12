# -*- coding: utf-8 -*-
"""
Least Squares Fitting Routine for Year 1 lab.
Lloyd Cawthorne 12/06/20

Adapted from lsfr.py by Abie Marshall 2016
    Adapted from LSFR.m credits to:
    Jordan Hulme
    Adam Petrus
    Ian Duerdoth
    Paddy Leahy

Reads in and validates data assuming given in columns: independent variable,
dependent variable, uncertainty on dependent variable.

Data is rejected and flagged for either being non-numerical or having an
uncertainty less than or equal to zero. The script continues with the accepted
data.

Performs a least squares fit to a linear function, produces reduced chi
squared, uncertainties and plot.

Details of the fitting procedure can be found here:
https://mathworld.wolfram.com/LeastSquaresFitting.html

You will learn how to write code similar to this in PHYS10362: Introduction
to Programming.

Edit the global constants (written in ALL_CAPS, below the import statements)
to edit the file input and plot attributes.

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

imageOnly = sys.argv[2] if len(sys.argv) > 2 else False
print(f'running: {sys.argv[1]}')
# File reading details
FILE_NAME = sys.argv[1]  # Remember to include the extension
SKIP_FIRST_LINE = True
DELIMITER = ','  # Set to space, ' ', if working with .txt file without commas

basename = os.path.basename(FILE_NAME)
PLOT_TITLE = f'Pressure Decay Fit for {basename}'
X_LABEL = 'Time / s'
Y_LABEL = r'$\ln\left(\frac{P_{in} - P_{out}}{P_{in} + P_{out}}\right)$'

AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]  # Not used unless AUTO_X_LIMITS = False
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]  # Not used unless AUTO_Y_LIMITS = False
LINE_COLOUR = 'red'  # See documentation for options:
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
LINE_STYLE = '-'   # See documentation for options:
# https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
MARKER_STYLE_LAMINAR = 'x'
MARKER_COLOUR_LAMINAR = 'black'
MARKER_STYLE_TURBULENT = 'o'
MARKER_COLOUR_TURBULENT = 'grey'
GRID_LINES = True
SAVE_FIGURE = True
FIGURE_NAME = FILE_NAME.replace('.csv', '.png')
if FIGURE_NAME == FILE_NAME: FIGURE_NAME += '.png'
FIGURE_RESOLUTION = 400  # in dpi


def linear_function(x_variable, parameters):
    """Outputs the result of y = mx + c, where m is the slope and c is the
    offset.  parameters is an array such that [slope, offset].
    Args:
        x_variable: float
        parameters: [float, float]
    Returns: float
    """
    return parameters[0] * x_variable + parameters[1]


def check_numeric(entry):
    """Checks if entry is numeric
    Args:
        entry: string
    Returns:
        bool
    Raises:
        ValueError: if entry cannot be cast to float type
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False


def check_uncertainty(uncertainty):
    """Checks uncertainty is non-zero and positive.
    Args:
        uncertainty: float
    Returns:
        Bool
    """
    if uncertainty > 0:
        return True
    return False


def validate_line(line):
    """Validates line. Outputs error messages accordingly.
    Args:
        line: string
    Returns:
        bool, if validation has been succesful
        line_floats, numpy array of floats
    """
    line_split = line.split(DELIMITER)

    if len(line_split) < 3:
        return False, None

    for entry in line_split:
        if check_numeric(entry) is False:
            print('Line omitted: {0:s}.'.format(line.strip('\n')))
            print('{0:s} is nonnumerical.'.format(entry))
            return False, None
    
    # We expect 4 columns: x, y, y_err, is_laminar
    # If only 3 are provided, we assume all are laminar (fallback)
    if len(line_split) == 3:
        line_floats = np.array([float(line_split[0]), float(line_split[1]),
                                float(line_split[2]), 1.0])
    else:
        line_floats = np.array([float(line_split[0]), float(line_split[1]),
                                float(line_split[2]), float(line_split[3])])
        
    if line_floats[2] <= 0:
        print('Line omitted: {0:s}.'.format(line.strip('\n')))
        print('Uncertainty must be greater than zero.')
        return False, None
    return True, line_floats


def open_file(file_name=FILE_NAME, skip_first_line=SKIP_FIRST_LINE):
    """Opens file, reads data and outputs data in numpy arrays.
    Args:
        file_name: string, default given by FILE_NAME
    Returns:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
        is_laminar: numpy array of floats (0 or 1)
    """
    x_data = np.array([])
    y_data = np.array([])
    y_uncertainties = np.array([])
    is_laminar = np.array([])
    
    try:
        raw_file_data = open(file_name, 'r', encoding='utf-8-sig')
    except FileNotFoundError:
        print("File '{0:s}' cannot be found.".format(file_name))
        return x_data, y_data, y_uncertainties, is_laminar
    
    for line in raw_file_data:
        if skip_first_line:
            skip_first_line = False
        else:
            line_valid, line_data = validate_line(line)
            if line_valid:
                x_data = np.append(x_data, line_data[0])
                y_data = np.append(y_data, line_data[1])
                y_uncertainties = np.append(y_uncertainties, line_data[2])
                is_laminar = np.append(is_laminar, line_data[3])
    raw_file_data.close()
    return x_data, y_data, y_uncertainties, is_laminar


def fitting_procedure(x_data, y_data, y_uncertainties):
    """Implements an analytic approach according to source in header."""
    weights = 1. / y_uncertainties**2
    repeated_term = (np.sum(weights) * np.sum(x_data**2 * weights)
                     - np.sum(x_data * weights)**2)
    slope = ((np.sum(weights) * np.sum(x_data * y_data * weights)
              - np.sum(x_data * weights) * np.sum(y_data * weights))
             / repeated_term)
    slope_uncertainty = np.sqrt(np.sum(weights) / repeated_term)
    offset = ((np.sum(y_data * weights) * np.sum(x_data**2 * weights)
               - np.sum(x_data * weights) * np.sum(x_data * y_data * weights))
              / repeated_term)
    offset_uncertainty = np.sqrt(np.sum(x_data**2 * weights) / repeated_term)

    return (np.array([slope, offset]), np.array([slope_uncertainty,
                                                 offset_uncertainty]))


def chi_squared_function(x_data, y_data, y_uncertainties, parameters):
    """Calculates the chi squared for the data given."""
    return np.sum((linear_function(x_data, [parameters[0], parameters[1]])
                   - y_data)**2 / y_uncertainties**2)


def create_plot(x_data, y_data, y_uncertainties, is_laminar, parameters,
                parameter_uncertainties):
    """Produces graphic of resulting fit"""
    figure = plt.figure(figsize=(8, 8), facecolor='white')
    
    # Use a grid to remove the gap between plots
    gs = figure.add_gridspec(4, 1)
    axes_main_plot = figure.add_subplot(gs[0:3, 0])

    # Plot Turbulent data
    mask_turbulent = (is_laminar == 0)
    if np.any(mask_turbulent):
        axes_main_plot.errorbar(x_data[mask_turbulent], y_data[mask_turbulent], 
                                yerr=y_uncertainties[mask_turbulent],
                                fmt=MARKER_STYLE_TURBULENT, color=MARKER_COLOUR_TURBULENT,
                                label='Excluded (Turbulent/Tail)', alpha=0.5)

    # Plot Laminar data
    mask_laminar = (is_laminar == 1)
    if np.any(mask_laminar):
        axes_main_plot.errorbar(x_data[mask_laminar], y_data[mask_laminar], 
                                yerr=y_uncertainties[mask_laminar],
                                fmt=MARKER_STYLE_LAMINAR, color=MARKER_COLOUR_LAMINAR,
                                label='Laminar Flow (Linear Fit)')

    # Fit line (across all x)
    x_fit_line = np.linspace(np.min(x_data), np.max(x_data), 100)
    axes_main_plot.plot(x_fit_line, linear_function(x_fit_line, parameters),
                        color=LINE_COLOUR, label='Linear Fit (Laminar only)')
    
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    axes_main_plot.set_ylabel(Y_LABEL)
    axes_main_plot.legend()

    # Fitting details
    x_fit = x_data[mask_laminar]
    y_fit = y_data[mask_laminar]
    y_err_fit = y_uncertainties[mask_laminar]
    
    chi_squared = chi_squared_function(x_fit, y_fit, y_err_fit, parameters)
    degrees_of_freedom = len(x_fit) - 2
    reduced_chi_squared = chi_squared / degrees_of_freedom

    # Statistics aligned to the right
    axes_main_plot.annotate(r'$\chi^2$ = {0:4.2f}'.format(chi_squared), (1, 0), (-5, -35),
                            xycoords='axes fraction', va='top', ha='right', textcoords='offset points', fontsize='10')
    axes_main_plot.annotate('d.o.f. = {0:d}'.format(degrees_of_freedom), (1, 0), (-5, -55),
                            xycoords='axes fraction', va='top', ha='right', textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(r'Red. $\chi^2$ = {0:4.2f}'.format(reduced_chi_squared), (1, 0), (-5, -70),
                            xycoords='axes fraction', va='top', ha='right', textcoords='offset points', fontsize='10')
    
    # Fit details aligned to the left
    axes_main_plot.annotate('Fit: $y=mx+c$', (0, 0), (5, -35),
                            xycoords='axes fraction', va='top', ha='left', textcoords='offset points')
    axes_main_plot.annotate('m = {0:6.4e} ± {1:6.4e}'.format(parameters[0], parameter_uncertainties[0]), (0, 0),
                            (5, -55), xycoords='axes fraction', va='top', ha='left', textcoords='offset points', fontsize='10')
    axes_main_plot.annotate('c = {0:6.4e} ± {1:6.4e}'.format(parameters[1], parameter_uncertainties[1]), (0, 0),
                            (5, -70), xycoords='axes fraction', va='top', ha='left', textcoords='offset points', fontsize='10')

    # Residuals plot
    residuals = y_data - linear_function(x_data, parameters)
    axes_residuals = figure.add_subplot(gs[3, 0], sharex=axes_main_plot)
    
    if np.any(mask_turbulent):
        axes_residuals.errorbar(x_data[mask_turbulent], residuals[mask_turbulent], 
                                yerr=y_uncertainties[mask_turbulent],
                                fmt=MARKER_STYLE_TURBULENT, color=MARKER_COLOUR_TURBULENT, alpha=0.5)
    
    if np.any(mask_laminar):
        axes_residuals.errorbar(x_data[mask_laminar], residuals[mask_laminar], 
                                yerr=y_uncertainties[mask_laminar],
                                fmt=MARKER_STYLE_LAMINAR, color=MARKER_COLOUR_LAMINAR)
        
    axes_residuals.plot(x_data, 0 * x_data, color=LINE_COLOUR)
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14)
    axes_residuals.set_xlabel(X_LABEL)
    axes_residuals.set_ylabel('Data - Fit')

    plt.tight_layout()
    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, dpi=FIGURE_RESOLUTION, transparent=False)
    if imageOnly:
        plt.close()
        return None
    plt.show()
    return None


def main():
    x_data, y_data, y_uncertainties, is_laminar = open_file()
    if len(x_data) == 0:
        return

    # Filter for fitting
    mask_laminar = (is_laminar == 1)
    if not np.any(mask_laminar):
        print("Warning: No laminar data found. Fitting to all data.")
        mask_laminar = np.ones(len(x_data), dtype=bool)

    x_fit = x_data[mask_laminar]
    y_fit = y_data[mask_laminar]
    y_err_fit = y_uncertainties[mask_laminar]

    parameters, parameter_uncertainties = fitting_procedure(x_fit, y_fit, y_err_fit)
    
    create_plot(x_data, y_data, y_uncertainties, is_laminar, parameters, parameter_uncertainties)


if __name__ == '__main__':
    main()
