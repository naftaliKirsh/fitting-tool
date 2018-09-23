import os
import traceback
import numpy as np
from multiprocessing import Pool, Value
import experiment as exp


counter = Value('i', 0)

def run(experiment, root, resonances, pwrs, meas='', if_band=1, num_pts=1, widths='',  waits=None, avg_number=None, ip_addr='169.254.252.66'):
    global counter
    processes=len(resonances) * len(pwrs)
    pool = Pool(processes=processes, initializer=exp.init_process, initargs=(counter, ))

    for power in pwrs:
        print "setting power to " + str(power) + " dBm"
        for resonance in resonances:
            print "setting center to " + str(resonance)
            freq = np.loadtxt('C:\\Users\\idomo\\PycharmProjects\\fitting-tool\\fake device\\' +str(power)+'_'+str(resonance)+'_freq.out', delimiter=',')
            print "saving frequencies to file"
            dataRaw = np.loadtxt('C:\\Users\\idomo\\PycharmProjects\\fitting-tool\\fake device\\'+str(power)+'_'+str(resonance)+'_data.out', delimiter=',')
            data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]
            experiment.make_fit(pool, freq, data, root, [str(resonance), str(power)], counter)
            print "saving data to file"
    return processes