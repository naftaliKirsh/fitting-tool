from unittest import TestCase
from resonator_tools import circuit
import skewed_lorentzian_fitter as slf
import numpy as np
import os


test_directory_path = r'C:\Users\Owner\PycharmProjects\fitting-tool\tests'
directory = os.listdir(test_directory_path)
tests = []
for file in directory:
    if file.endswith('_freq.out'):
        tests.append(file.split('_')[0])


class TestFit(TestCase):
    def test_fit(self):
        for test in tests:
            FreqFile = os.path.join(test_directory_path,test+'_freq.out')
            DataFile = os.path.join(test_directory_path,test+'_data.out')
            try:
                freq = np.loadtxt(FreqFile, delimiter=',')
                dataRaw = np.loadtxt(DataFile, delimiter=',')
            except IOError:
                print 'ERROR: file not found'
                exit(1)
            data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]
            port1 = circuit.notch_port()
            port1.add_data(freq, data)
            assert slf.fit(port1)
