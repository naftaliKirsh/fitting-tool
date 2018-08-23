import numpy as np
import matplotlib.pyplot as plt
import lmfit
from resonator_tools import circuit

DATA_FILE = r'C:\Users\Owner\PycharmProjects\fitting-tool\temp\LORENTZIAN_data.out'
FREQ_FILE = r'C:\Users\Owner\PycharmProjects\fitting-tool\temp\LORENTZIAN_freq.out'


def residuals(params, x, y):
    # A2 = params['A2']
    A4 = params['A4']
    Ql = params['Ql']
    # err = y - (A1a + A2 * (x - fra) + (A3a + A4 * (x - fra)) / (1. + 4. * Ql ** 2 * ((x - fra) / fra) ** 2))
    err = y - (A1a + (A3a + A4 * (x - fra)) / (1. + 4. * Ql ** 2 * ((x - fra) / fra) ** 2))
    return err


try:
    freq = np.loadtxt(FREQ_FILE, delimiter=',')
    dataRaw = np.loadtxt(DATA_FILE, delimiter=',')
except IOError:
    print 'ERROR: file not found'
    exit(1)
data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]

port1 = circuit.notch_port()
port1.add_data(freq, data)

z_data_normalized =  port1.z_data_raw / np.max(np.absolute(port1.z_data_raw))
plt.plot(port1.f_data, z_data_normalized.real)

amplitude = np.absolute(z_data_normalized)
amplitude_sqr = amplitude ** 2
A1a = np.minimum(amplitude_sqr[0], amplitude_sqr[-1])
A3a = np.min(amplitude_sqr)
fra = port1.f_data[np.argmin(amplitude_sqr)]

params = lmfit.Parameters()
minimum = (port1.f_data[np.argmin(z_data_normalized.real)],np.min(z_data_normalized.real))
maximum = (port1.f_data[np.argmax(z_data_normalized.real)],np.max(z_data_normalized.real))
left_tail_f_data = port1.f_data[0 : np.argmin(z_data_normalized.real)]
left_tail_z_data = z_data_normalized[0 : np.argmin(z_data_normalized.real)]

right_tail_f_data = port1.f_data[np.argmax(z_data_normalized.real):-1]
right_tail_z_data = z_data_normalized[np.argmax(z_data_normalized.real):-1]

plt.figure(2)
# plt.plot(right_tail_f_data, right_tail_z_data)
plt.plot(left_tail_f_data, left_tail_z_data)

mod = lmfit.models.ExpressionModel('(a/(x-c))+d', independent_vars=['x'])

params = mod.make_params()
# params= lmfit.minimize(mod,params).params.valuesdict().values()
out = mod.fit(left_tail_z_data, params, x=left_tail_f_data)
a,c,d = out.params.values()
x = np.linspace(-1050000,0,1e5)
plt.plot(x, (a/(x-c))+d, '--')
plt.show()

params.add('Ala', A1a, min=np.minimum(z_data_normalized.real[0], z_data_normalized.real[-1]), max=np.maximum(z_data_normalized.real[0], z_data_normalized.real[-1]))
params.add('A3a', value=(np.min(z_data_normalized.real-np.average((z_data_normalized.real[0], z_data_normalized.real[-1])))+np.max(z_data_normalized.real-np.average((z_data_normalized.real[0], z_data_normalized.real[-1])))))
params.add('fra', fra, min=fra-0.5e5, max=fra+0.5e5)
# params.add('A2', value=0.0, vary=False) #     GET
params.add('Ql', value=1e3) #                           GUESSES
params.add('A4', value=0.0) #                     BETTER

x = port1.f_data
out = lmfit.minimize(residuals, params, args=(x, z_data_normalized.real))
A1, A3, fr, Ql, A4 = out.params.valuesdict().values()
plt.plot(x, (A1 + (A3 + A4 * (x - fr)) / (1. + 4. * Ql ** 2 * ((x - fr) / fr) ** 2)))
plt.legend(['raw data', 'initial values'])
plt.show(block=False)
pass
