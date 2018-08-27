import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
from resonator_tools import circuit

DATA_FILE = r'C:\Users\Owner\PycharmProjects\fitting-tool\temp\LORENTZIAN_data.out'
FREQ_FILE = r'C:\Users\Owner\PycharmProjects\fitting-tool\temp\LORENTZIAN_freq.out'


def find_nearest_neighbor(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_slope(x_data=list, y_data=list, point=(None,None)):
    try:
        point_index = x_data.index(point[0])
    except:
        raise IndexError('the inquierd poin is not in the data')
    gradient = np.gradient(z_data_normalized.real)
    gradient[point_index]
    slope = 1
    return slope

def residuals(p, x, y):
    A2, A4, Ql = p
    err = y - (A1a + A2 * (x - fra) + (A3a + A4 * (x - fra)) / (1. + 4. * Ql ** 2 * ((x - fra) / fra) ** 2))
    return err


def fitfunc(x, A1, A2, A3, A4, fr, Ql):
    return A1 + 0 * (x - fr) + (A3 + A4 * (x - fr)) / (1. + 4. * Ql ** 2 * ((
                                                                                    x - fr) / fr) ** 2)  # TODO: change 0 to A2 and fit A2 using a averaged weighted linear fit of the tail ends then average the slopes


def a_over_x(x, a, c, d):
    return (a / (x - c)) + d


def const(x, c):
    return c


def linear(x, a, b):
    return a + b * x


try:
    freq = np.loadtxt(FREQ_FILE, delimiter=',')
    dataRaw = np.loadtxt(DATA_FILE, delimiter=',')
except IOError:
    print 'ERROR: file not found'
    exit(1)

data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]

port1 = circuit.notch_port()
port1.add_data(freq, data)

z_data_normalized = port1.z_data_raw / np.max(np.absolute(port1.z_data_raw))
plt.plot(port1.f_data, z_data_normalized.real)

amplitude = np.absolute(z_data_normalized)
amplitude_sqr = amplitude ** 2
A1a = np.minimum(amplitude_sqr[0], amplitude_sqr[-1])
A3a = np.min(amplitude_sqr)
fra = port1.f_data[np.argmin(amplitude_sqr)]

minimum = (port1.f_data[np.argmin(z_data_normalized.real)], np.min(z_data_normalized.real))
maximum = (port1.f_data[np.argmax(z_data_normalized.real)], np.max(z_data_normalized.real))
left_tail_f_data = port1.f_data[0: np.argmin(z_data_normalized.real)]
left_tail_z_data = z_data_normalized[0: np.argmin(z_data_normalized.real)]

right_tail_f_data = port1.f_data[np.argmax(z_data_normalized.real):-1]
right_tail_z_data = z_data_normalized[np.argmax(z_data_normalized.real):-1]

plt.figure(2)
plt.plot(right_tail_f_data, right_tail_z_data.real)
plt.plot(left_tail_f_data, left_tail_z_data.real)

resultsL = scopt.curve_fit(a_over_x, left_tail_f_data, left_tail_z_data.real, p0=[1, minimum[0], A1a])
errL = np.sqrt(np.diag(resultsL[1]))
plt.plot(left_tail_f_data, a_over_x(left_tail_f_data, resultsL[0][0], resultsL[0][1], resultsL[0][2]))

resultsR = scopt.curve_fit(a_over_x, right_tail_f_data, right_tail_z_data.real,
                           p0=[1, maximum[0], z_data_normalized.real[-1]])
errR = np.sqrt(np.diag(resultsR[1]))
plt.plot(right_tail_f_data, a_over_x(right_tail_f_data, resultsR[0][0], resultsR[0][1], resultsR[0][2]))

if errL[2] < errR[2]:
    results = resultsL
    err = errL
else:
    results = resultsR
    err = errR


def A3(A1):
    return minimum[1] + maximum[1] - 2 * A1


A1 = results[0][2]
A3 = A3(A1)
magic_point_nearest_neighbor_index = find_nearest_neighbor(z_data_normalized.real, A1 + A3)
if maximum[0] > minimum[0]:
    fr = port1.f_data[z_data_normalized.real.tolist().index(magic_point_nearest_neighbor_index, -1)]
else:
    fr = port1.f_data[z_data_normalized.real.tolist().index(magic_point_nearest_neighbor_index, 0)]


initial_values = [results[0][2], 0, A3(results[0][2]), 1, minimum[0], 1]
fit_results = scopt.curve_fit(fitfunc, port1.f_data, z_data_normalized.real, p0=initial_values,
                              bounds=([np.minimum(z_data_normalized.real[0], z_data_normalized.real[-1]), -np.inf,
                                       np.minimum(A3(z_data_normalized.real[0]), A3(z_data_normalized.real[-1])),
                                       -np.inf, minimum[0], -np.inf],
                                      [np.maximum(z_data_normalized.real[0], z_data_normalized.real[-1]), np.inf,
                                       np.maximum(A3(z_data_normalized.real[0]), A3(z_data_normalized.real[-1])),
                                       np.inf, maximum[0], np.inf]))

plt.figure(1)
plt.plot(port1.f_data, fitfunc(port1.f_data, fit_results[0][0], fit_results[0][1], fit_results[0][2], fit_results[0][3],
                               fit_results[0][4], fit_results[0][5]))
plt.show(block=False)
exit(0)
