import pytest
from resonator_tools import circuit
import fitting_tool
import numpy as np
import os

test_directory_path = r'C:\Users\Owner\PycharmProjects\fitting-tool\tests'
directory = os.listdir(test_directory_path)
tests = []
for file in directory:
    if file.endswith('_freq.out'):
        tests.append(file.split('_')[0])
test_parameters = []
for test in tests:
    FreqFile = os.path.join(test_directory_path, test + '_freq.out')
    DataFile = os.path.join(test_directory_path, test + '_data.out')
    try:
        freq = np.loadtxt(FreqFile, delimiter=',')
        dataRaw = np.loadtxt(DataFile, delimiter=',')
    except IOError:
        print 'ERROR: file not found'
        exit(1)
    data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]
    port1 = circuit.notch_port()
    port1.add_data(freq, data)
    with open(os.path.join(test_directory_path, test + '_params.out'), 'r') as fo:
        expected_results = fo.read()
    results_list = eval(expected_results)
    test_parameters.append((port1.f_data, port1.z_data_raw, [results_list[3][1], results_list[0][1], results_list[6][1]], test))
print ''

@pytest.mark.parametrize('f_data, z_data, fit_results, test', test_parameters)
def test_fit(f_data, z_data, fit_results, test):
    port1 = circuit.notch_port()
    port1.add_data(f_data, z_data)
    delay, results = fitting_tool.fit(port1)
    assert fit_results[0] == pytest.approx(results[4],abs=0.02*results[4])\
           and fit_results[1] == pytest.approx(results[5],0.02*results[5])\
           and fit_results[2] == pytest.approx(delay, 0.02*delay), test