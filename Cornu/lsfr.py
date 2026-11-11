# -*- coding: utf-8 -*-
"""
Least Squares Fitting Routine plotting module for Cornu's Method lab.
Adapted from LSFR-25.py (Lloyd Cawthorne / Abie Marshall).
"""

import numpy as np
import matplotlib.pyplot as plt

AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]
LINE_COLOUR = 'red'
LINE_STYLE = '-'
MARKER_STYLE = 'x'
MARKER_COLOUR = 'black'
GRID_LINES = True
SAVE_FIGURE = True
FIGURE_RESOLUTION = 400  # in dpi


def linear_function(x_variable, parameters):
    """Outputs the result of y = mx + c, where m is the slope and c is the
    offset. parameters is an array such that [slope, offset].
    """
    return parameters[0] * x_variable + parameters[1]


def chi_squared_function(x_data, y_data, y_uncertainties, parameters):
    """Calculates the chi squared for the data given, assuming a linear
    relationship.
    """
    return np.sum((linear_function(x_data, [parameters[0], parameters[1]])
                   - y_data)**2 / y_uncertainties**2)


def create_plot(x_data, y_data, parameters, parameter_uncertainties,
                y_uncertainties=None, x_uncertainties=None, r_squared=None,
                plot_title=r'Fringe Order $n$ vs $x_n^2$',
                x_label=r'$x_n^2$ / $\mathrm{mm}^2$',
                y_label=r'Fringe Order $n$',
                figure_name='lsfr_plot.png',
                image_only=True):
    """Produces graphic of resulting linear fit and residuals in LSFR style."""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if x_uncertainties is not None:
        x_uncertainties = np.asarray(x_uncertainties, dtype=float)
    if y_uncertainties is not None:
        y_uncertainties = np.asarray(y_uncertainties, dtype=float)

    if y_uncertainties is not None and x_uncertainties is not None:
        sigma_eff = np.sqrt(y_uncertainties**2 + (parameters[0] * x_uncertainties)**2)
    elif y_uncertainties is not None:
        sigma_eff = y_uncertainties
    elif x_uncertainties is not None:
        sigma_eff = np.abs(parameters[0] * x_uncertainties)
    else:
        sigma_eff = None

    figure = plt.figure(figsize=(8, 6), facecolor='white')

    axes_main_plot = figure.add_subplot(211)

    if y_uncertainties is not None or x_uncertainties is not None:
        axes_main_plot.errorbar(x_data, y_data, xerr=x_uncertainties,
                                yerr=y_uncertainties,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
    else:
        axes_main_plot.plot(x_data, y_data, MARKER_STYLE, color=MARKER_COLOUR)

    axes_main_plot.plot(x_data, linear_function(x_data, parameters),
                        linestyle=LINE_STYLE, color=LINE_COLOUR)
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.set_title(plot_title, fontsize=14)
    axes_main_plot.set_xlabel(x_label)
    axes_main_plot.set_ylabel(y_label)

    degrees_of_freedom = len(x_data) - 2

    if sigma_eff is not None:
        chi_squared = chi_squared_function(x_data, y_data, sigma_eff,
                                           parameters)
        reduced_chi_squared = chi_squared / degrees_of_freedom
        axes_main_plot.annotate((r'$\chi^2$ = {0:4.2f}'.
                                 format(chi_squared)), (1, 0), (-60, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                                 format(degrees_of_freedom)), (1, 0), (-147, -55),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate((r'Reduced $\chi^2$ = {0:4.2f}'.
                                 format(reduced_chi_squared)), (1, 0), (-104, -70),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
    else:
        if r_squared is None:
            ss_res = np.sum((y_data - linear_function(x_data, parameters))**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        axes_main_plot.annotate((r'$R^2$ = {0:6.4f}'.
                                 format(r_squared)), (1, 0), (-85, -35),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')
        axes_main_plot.annotate(('Degrees of freedom = {0:d}'.
                                 format(degrees_of_freedom)), (1, 0), (-147, -55),
                                xycoords='axes fraction', va='top',
                                textcoords='offset points', fontsize='10')

    axes_main_plot.annotate('Fit: $y=mx+c$', (0, 0), (0, -35),
                            xycoords='axes fraction', va='top',
                            textcoords='offset points')
    axes_main_plot.annotate(('m = {0:6.4e}'.format(parameters[0])), (0, 0),
                            (0, -55), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[0])),
                            (0, 0), (100, -55), xycoords='axes fraction',
                            va='top', fontsize='10',
                            textcoords='offset points')
    axes_main_plot.annotate(('c = {0:6.4e}'.format(parameters[1])), (0, 0),
                            (0, -70), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='10')
    axes_main_plot.annotate(('± {0:6.4e}'.format(parameter_uncertainties[1])),
                            (0, 0), (100, -70), xycoords='axes fraction',
                            textcoords='offset points', va='top',
                            fontsize='10')

    # Residuals plot
    residuals = y_data - linear_function(x_data, parameters)
    axes_residuals = figure.add_subplot(414)
    if sigma_eff is not None:
        axes_residuals.errorbar(x_data, residuals, yerr=sigma_eff,
                                fmt=MARKER_STYLE, color=MARKER_COLOUR)
    else:
        axes_residuals.plot(x_data, residuals, MARKER_STYLE, color=MARKER_COLOUR)
    axes_residuals.plot(x_data, 0 * x_data, color=LINE_COLOUR)
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14)

    if not AUTO_X_LIMITS:
        axes_main_plot.set_xlim(X_LIMITS)
        axes_residuals.set_xlim(X_LIMITS)
    if not AUTO_Y_LIMITS:
        axes_main_plot.set_ylim(Y_LIMITS)
        axes_residuals.set_ylim(Y_LIMITS)

    if SAVE_FIGURE:
        plt.savefig(figure_name, dpi=FIGURE_RESOLUTION, transparent=False)
    if image_only:
        plt.close(figure)
        return None
    plt.show()
    return None
