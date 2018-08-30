import numpy as np
import scipy.optimize as scopt
import scipy.stats as stats
from resonator_tools import circuit
from scipy.interpolate import UnivariateSpline


import Tkinter, tkFileDialog
import matplotlib.pyplot as plt

def file_chooser(format):
    root = Tkinter.Tk()
    root.withdraw()  # hide root
    root.File = tkFileDialog.askopenfilename(filetypes=(("data files", format), ("all files", "*.*")))
    root.deiconify()
    root.destroy()
    return root.File

def find_nearest_neighbor(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_slope(x_data=list, y_data=list, point=(None, None)):
    try:
        point_index = x_data.index(point[0])
    except:
        raise IndexError('the inquierd poin is not in the data')
    radius = np.abs(y_data.index(np.min(y_data)) - point_index)
    linear_regression_fit = stats.linregress(x_data[point_index - radius:point_index + radius + 1],
                                             y_data[point_index - radius:point_index + radius + 1])
    return linear_regression_fit.slope


def residuals(p, x, y):
    A2, A4, Ql = p
    err = y - (A1a + A2 * (x - fra) + (A3a + A4 * (x - fra)) / (1. + 4. * Ql ** 2 * ((x - fra) / fra) ** 2))
    return err


def fitfunc(x, A1, A2, A3, A4, fr, Ql):
    return A1 + A2 * (x - fr) + (A3 + A4 * (x - fr)) / (1. + 4. * Ql ** 2 * ((
                                                                                     x - fr) / fr) ** 2)  # TODO: change 0 to A2 and fit A2 using a averaged weighted linear fit of the tail ends then average the slopes


def a_over_x(x, a, c, d):
    return (a / (x - c)) + d


def const(x, c):
    return c


def linear(x, a, b):
    return a + b * x


def fit(port1): #TODO: reduce randomness of fit parameters in noisy fit curreent approx(0.04)
    z_data_normalized_original = port1.z_data_raw / np.max(np.absolute(port1.z_data_raw))
    linreg = stats.linregress(port1.f_data, z_data_normalized_original.real)
    z_data_stratened = z_data_normalized_original.real - linreg.slope * port1.f_data - linreg.intercept
    dist = np.abs(z_data_normalized_original[list(z_data_stratened).index(np.min(z_data_stratened))].real - np.min(
        z_data_stratened.real))
    z_data_normalized_stratened = z_data_stratened + dist

    amplitude = np.absolute(z_data_normalized_stratened)
    amplitude_sqr = amplitude ** 2
    global A1a
    A1a = np.minimum(amplitude_sqr[0], amplitude_sqr[-1])
    global A3a
    A3a = np.min(amplitude_sqr)
    global fra
    fra = port1.f_data[np.argmin(amplitude_sqr)]

    minimum = (port1.f_data[np.argmin(z_data_normalized_stratened.real)], np.min(z_data_normalized_stratened.real))
    maximum = (port1.f_data[np.argmax(z_data_normalized_stratened.real)], np.max(z_data_normalized_stratened.real))
    left_tail_f_data = port1.f_data[0: np.argmin(z_data_normalized_stratened.real)]
    left_tail_z_data = z_data_normalized_stratened[0: np.argmin(z_data_normalized_stratened.real)]

    right_tail_f_data = port1.f_data[np.argmax(z_data_normalized_stratened.real):-1]
    right_tail_z_data = z_data_normalized_stratened[np.argmax(z_data_normalized_stratened.real):-1]

    resultsL = scopt.curve_fit(a_over_x, left_tail_f_data, left_tail_z_data.real,
                               p0=[1, minimum[0], np.random.uniform(np.minimum(z_data_normalized_stratened.real[0],
                                                                               z_data_normalized_stratened.real[-1]),
                                                                    np.maximum(z_data_normalized_stratened.real[0],
                                                                               z_data_normalized_stratened.real[-1]),
                                                                    1)],
                               bounds=([-np.inf, -np.inf, np.minimum(z_data_normalized_stratened.real[0],
                                                                     z_data_normalized_stratened.real[-1])],
                                       [np.inf, np.inf, np.maximum(z_data_normalized_stratened.real[0],
                                                                   z_data_normalized_stratened.real[-1])]))
    errL = np.sqrt(np.diag(resultsL[1]))
    try:
        resultsR = scopt.curve_fit(a_over_x, right_tail_f_data, right_tail_z_data.real,
                                   p0=[1, maximum[0], np.random.uniform(np.minimum(z_data_normalized_stratened.real[0],
                                                                                   z_data_normalized_stratened.real[
                                                                                       -1]),
                                                                        np.maximum(z_data_normalized_stratened.real[0],
                                                                                   z_data_normalized_stratened.real[
                                                                                       -1]), 1)],
                                   bounds=([-np.inf, -np.inf, np.minimum(z_data_normalized_stratened.real[0],
                                                                         z_data_normalized_stratened.real[-1])],
                                           [np.inf, np.inf, np.maximum(z_data_normalized_stratened.real[0],
                                                                       z_data_normalized_stratened.real[-1])]))
        errR = np.sqrt(np.diag(resultsR[1]))
    except ValueError:
        errR = [None, None, None]
        errR[2] = np.inf

    if errL[2] <= errR[2]:
        results = resultsL
        err = errL
    else:
        results = resultsR
        err = errR

    def _A3(A1):
        return minimum[1] + maximum[1] - 2 * A1

    A1 = results[0][2]
    A2 = linreg.slope
    A3 = _A3(A1)
    if maximum[0] > minimum[0]:
        if A3 > 0:
            idx = find_nearest_neighbor(z_data_normalized_stratened.real[0:list(port1.f_data).index(maximum[0])],
                                        A1 + A3)
            fr = port1.f_data[idx]
        else:
            idx = find_nearest_neighbor(z_data_normalized_stratened.real[list(port1.f_data).index(minimum[0]):],
                                        A1 + A3) + len(
                z_data_normalized_stratened.real[0:list(port1.f_data).index(minimum[0])]) + 1
            fr = port1.f_data[idx]
    else:
        if A3 > 0:
            idx = find_nearest_neighbor(z_data_normalized_stratened.real[list(port1.f_data).index(maximum[0]):],
                                        A1 + A3) + len(
                z_data_normalized_stratened.real[0:list(port1.f_data).index(maximum[0])]) + 1
            fr = port1.f_data[idx]
        else:
            idx = find_nearest_neighbor(z_data_normalized_stratened.real[0:list(port1.f_data).index(minimum[0])],
                                        A1 + A3)
            fr = port1.f_data[idx]

    A4 = find_slope(list(port1.f_data), list(z_data_normalized_stratened.real), (fr, A1 + A3))
    if A3 < 0:
        spline = UnivariateSpline(port1.f_data, z_data_normalized_stratened.real - A1 + np.max(np.abs(
            z_data_normalized_stratened.real - A1)) * 0.5, s=0)
    else:
        spline = UnivariateSpline(port1.f_data, z_data_normalized_stratened.real - A1 - np.max(np.abs(
            z_data_normalized_stratened.real - A1)) * 0.5, s=0)
    roots = spline.roots()
    Ql = fr / np.abs(roots[1] - roots[0])

    # if not (np.maximum(minimum[0], maximum[0]) > fr and fr > np.minimum(minimum[0], maximum[0])):
    #     fr = np.random.uniform(np.minimum(minimum[0], maximum[0]), np.maximum(minimum[0], maximum[0]), 1)
    initial_values = [A1, 0, A3, A4, fr, Ql]
    popt, pconv = scopt.curve_fit(fitfunc, port1.f_data, z_data_normalized_original.real, p0=initial_values,
                                  bounds=([np.minimum(z_data_normalized_stratened.real[0],
                                                      z_data_normalized_stratened.real[-1]), -np.abs(A2),
                                           np.minimum(_A3(z_data_normalized_stratened.real[0]),
                                                      _A3(z_data_normalized_stratened.real[-1])),
                                           -np.inf, np.minimum(minimum[0], maximum[0]), 0],
                                          [np.maximum(z_data_normalized_stratened.real[0],
                                                      z_data_normalized_stratened.real[-1]), np.abs(A2),
                                           np.maximum(_A3(z_data_normalized_stratened.real[0]),
                                                      _A3(z_data_normalized_stratened.real[-1])),
                                           np.inf, np.maximum(minimum[0], maximum[0]), np.inf]))

    if pconv is not None:
        port1.df_error = np.sqrt(pconv[4][4])
        port1.dQl_error = np.sqrt(pconv[5][5])
    else:
        port1.df_error = np.inf
        port1.dQl_error = np.inf

    return popt

if __name__=='__main__':
    FreqFile = file_chooser('*_freq.out')
    DataFile = file_chooser('*_data.out')
    try:
        freq = np.loadtxt(FreqFile, delimiter=',')
        dataRaw = np.loadtxt(DataFile, delimiter=',')
    except IOError:
        print 'ERROR: file not found'
        exit(1)
    data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]
    port1 = circuit.notch_port()
    port1.add_data(freq, data)
    results = fit(port1)
    A1, A2, A3, A4, fr, Ql = results
    z_data_normalized_original = port1.z_data_raw / np.max(np.absolute(port1.z_data_raw))
    plt.plot(port1.f_data, z_data_normalized_original)
    plt.plot(port1.f_data, fitfunc(port1.f_data, A1, A2, A3, A4, fr, Ql))
    plt.legend(['generated', 'fitted'])
    plt.show()
