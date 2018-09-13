"""Changes in v5.7 : save all measurements of a resonance in a single file"""
import sys
sys.path.insert(0,r'X:\CodeVault\PythonLibs')
import N5230A.N5230A as nalib
from time import sleep
import os
from os.path import join
from scipy import linspace
import traceback
import numpy as np
from multiprocessing import Pool


resonances = ['121.375e6','66.475e6'] #in Hz
experiment_folder = r"Y:\Users\StudentProjects\edo mor\Measurement\experiment1" #must be a new folder
meas = "S44" #trace type
if_band = 1000.0 #IF bandwidth [Hz]
num_pts = 801.0 #number of points
averaging = False
avg_number = 60000  #more averages than time, so it'll not start from the beginning
widths = [9e6/1e3,5e6/1e3]#(span) in KHz (length equal to # of resonances)
pwrs =  linspace(-30,-20,2) #dBm
time_one_sweep = (num_pts/if_band)




def run(experiment, root, resonances, meas, if_band, num_pts, widths, pwrs, time_one_sweep=(num_pts/if_band), waits=None, avg_number=None, ip_addr='169.254.252.66'):

    pool = Pool(processes=len(resonances) * len(pwrs) * 2)

    if waits == None:
        waits = [3] * len(pwrs)

    time_wait= waits  # Time to measure at each power. in seconds. (length equal to # of powers)

    note = "PowerSweep on all resonators, including the doubtful ones"
    if len(pwrs) != len(time_wait):
        print "len(pwrs) should be equal to len(time_wait), exiting.."
        sys.exit(1)
    if len(resonances) != len(widths):
        print "len(resonances) should be equal to len(widths), exiting.."
        sys.exit(1)

    print "Expected time for measurement: " + str(sum(time_wait) * len(resonances) / 60.0) + " minutes"

    # connect
    na = nalib.N5230A(ip_addr)

    j = 0
    firstTime = True
    ctrlC = False

    try:
        # paramFile = open(join(experiment_folder, "paramsFile.txt"), "w")                 ##############################################################
        # paramFile.write("script: pwr_sweep_nums5.7.py\n")                                            R              R                   R             #
        # paramFile.write("IF bandwith: " + str(if_band) + " Hz\n")                                     E              E                   E            #
        # paramFile.write("Number of points: " + str(num_pts) + "\n")                                    M              M                   M           #
        # paramFile.write("Spans: " + str(widths) + " [kHz]\n")                                           O              O                   O          #
        # if averaging:                                                                                    V              V                   V         #
        #     paramFile.write("Averaging on. Maximal number of averages: " + str(avg_number) + "\n")        E              E                   E        #
        # else:                                                                                        R              R                   R             #
        #     paramFile.write("Averaging off.\n")                                                       E              E                   E            #
        # paramFile.write("Resonances scanned: " + str(resonances) + " Hz\n")                            M              M                   M           #
        # paramFile.write("Power levels scanned: " + str(pwrs) + " dBm\n")                                O              O                   O          #
        # paramFile.write("Time waited at each power level: " + str(time_wait) + " seconds\n")             V              V                   V         #
        # paramFile.write("Note:" + note + "\n")                                                            E              E                   E        #
        # paramFile.close()                                                                 #############################################################
        # create measurement
        na.sendCommand("CALC:PAR:DEL:ALL")  # delete old
        na.sendCommand("CALC:PAR:DEF M1," + meas)  # create new
        na.sendCommand("CALC:PAR:SEL M1")  # select it
        na.sendCommand("DISP:WIND:TRAC:FEED M1")  # show it
        na.sendCommand("SENS:BAND " + str(if_band))  # if band
        na.sendCommand("SENS:SWE:MODE CONT")  # Continuous mode
        na.sendCommand("SENS:SWE:POIN " + str(num_pts))  # number of points
        if avg_number!=None:
            na.sendCommand("SENS:AVER ON")  # averaging on
            na.sendCommand("SENS:AVER:COUN " + str(avg_number))  # number of averages
        else:
            na.sendCommand("SENS:AVER OFF")  # averaging off
        na.sendCommand("SENS:SWE:TIME:AUTO 1")  # automatic (minimal) sweep time
        for power in pwrs:
            na.sendCommand("CALC:PAR:SEL M1")  # select it
            print "setting power to " + str(power) + " dBm"
            na.sendCommand("SOUR:POW1 " + str(power))
            if firstTime:
                print "turning power on"
                na.sendCommand("OUTP ON")
                firstTime = False
            rIdx = 0
            for resonance in resonances:
                # outFile = open(join(experiment_folder, resonance + "_data.out"), "a")
                na.sendCommand("CALC:PAR:SEL M1")  # select it
                print "setting center to " + str(resonance)
                na.sendCommand("SENS:FREQ:CENT " + resonance)
                print "setting width to " + str(widths[rIdx] * 1000)
                na.sendCommand("SENS:FREQ:SPAN " + str(widths[rIdx] * 1000))
                if j == 0:  # first scan for this resonance
                    # freqFile = open(join(experiment_folder, resonance + "_freq.out"), "w")
                    freq = na.query("sens1:x?")
                    print "saving frequencies to file"
                    # freqFile.write(freq)
                    # freqFile.close()
                print "Waiting for " + str(time_wait[j]) + " for data"
                sleep(time_wait[j])
                data = na.query("calc:data? sdata")
                freq_num_str = freq.split(",")
                freq_num = np.array([float(x) for x in freq_num_str])
                data_str = data.split(",")
                dataRaw = np.array([float(x) for x in data_str])
                dataComplex = dataRaw[0::2] + 1j * dataRaw[1::2]

                experiment.make_fit(pool, freq_num, dataComplex, root, [resonance, power])

                print "saving data to file"
                # outFile.write(data + "\n")
                # outFile.flush()
                # # os.fsync(outFile.fileno()) - causes errors
                # outFile.close()
                rIdx = rIdx + 1
            j = j + 1

    except KeyboardInterrupt:
        print "interrupted by Ctrl-C"
        ctrlC = True
    except Exception, e:
        traceback.print_exc()
        print("\n\n")
        print e

    finally:  # clean-up
        print "setting power off and trigger hold"
        na.sendCommand("OUTP OFF")  # set power off at end for safety
        na.sendCommand("SENS:SWE:MODE HOLD")  # Hold mode

    if ctrlC:
        r = raw_input("Enter Y to delete saved data files...")
        if r == "Y":
            from shutil import rmtree
            rmtree(experiment_folder)

    na.close()
    a = raw_input('press enter to close window... \n')  # TODO: make a better stopping mechanisem


if __name__=='__main__':
    print 'dont run the script call it as a module'

    #
    # pool = Pool(processes=len(resonances) * len(pwrs) * 2)
    #
    # time_wait = [3]*2 #Time to measure at each power. in seconds. (length equal to # of powers)
    #
    # note = "PowerSweep on all resonators, including the doubtful ones"
    # if len(pwrs)!=len(time_wait):
    #     print "len(pwrs) should be equal to len(time_wait), exiting.."
    #     sys.exit(1)
    # if len(resonances)!=len(widths):
    #     print "len(resonances) should be equal to len(widths), exiting.."
    #     sys.exit(1)
    #
    # print "Expected time for measurement: "+str(sum(time_wait)*len(resonances)/60.0)+" minutes"
    # os.mkdir(experiment_folder)
    # if not os.path.exists(os.path.join(experiment_folder, 'results')):
    #     os.mkdir(os.path.join(experiment_folder, 'results'))
    #
    # #connect
    # na = nalib.N5230A("169.254.252.66")
    #
    # j=0
    # firstTime = True
    # ctrlC = False
    #
    #
    # try:
    #     paramFile = open(join(experiment_folder, "paramsFile.txt"), "w")
    #     paramFile.write("script: pwr_sweep_nums5.7.py\n")
    #     paramFile.write("IF bandwith: "+str(if_band)+" Hz\n")
    #     paramFile.write("Number of points: "+str(num_pts)+"\n")
    #     paramFile.write("Spans: "+str(widths)+" [kHz]\n")
    #     if averaging:
    #         paramFile.write("Averaging on. Maximal number of averages: "+str(avg_number)+"\n")
    #     else:
    #         paramFile.write("Averaging off.\n")
    #     paramFile.write("Resonances scanned: "+str(resonances)+" Hz\n")
    #     paramFile.write("Power levels scanned: "+str(pwrs)+" dBm\n")
    #     paramFile.write("Time waited at each power level: "+str(time_wait)+" seconds\n")
    #     paramFile.write("Note:"+note+"\n")
    #     paramFile.close()
    #     #create measurement
    #     na.sendCommand("CALC:PAR:DEL:ALL") #delete old
    #     na.sendCommand("CALC:PAR:DEF M1,"+meas) #create new
    #     na.sendCommand("CALC:PAR:SEL M1") #select it
    #     na.sendCommand("DISP:WIND:TRAC:FEED M1") #show it
    #     na.sendCommand("SENS:BAND "+str(if_band)) #if band
    #     na.sendCommand("SENS:SWE:MODE CONT") #Continuous mode
    #     na.sendCommand("SENS:SWE:POIN "+str(num_pts)) #number of points
    #     if averaging:
    #          na.sendCommand("SENS:AVER ON") #averaging on
    #          na.sendCommand("SENS:AVER:COUN "+str(avg_number)) #number of averages
    #     else:
    #         na.sendCommand("SENS:AVER OFF") #averaging off
    #     na.sendCommand("SENS:SWE:TIME:AUTO 1") #automatic (minimal) sweep time
    #     for power in pwrs:
    #             na.sendCommand("CALC:PAR:SEL M1") #select it
    #             print "setting power to "+str(power)+" dBm"
    #             na.sendCommand("SOUR:POW1 "+str(power))
    #             if firstTime:
    #                 print "turning power on"
    #                 na.sendCommand("OUTP ON")
    #                 firstTime = False
    #             rIdx = 0
    #             for resonance in resonances:
    #                 outFile = open(join(experiment_folder, resonance + "_data.out"), "a")
    #                 na.sendCommand("CALC:PAR:SEL M1") #select it
    #                 print "setting center to "+str(resonance)
    #                 na.sendCommand("SENS:FREQ:CENT "+resonance)
    #                 print "setting width to "+str(widths[rIdx]*1000)
    #                 na.sendCommand("SENS:FREQ:SPAN "+str(widths[rIdx]*1000))
    #                 if j==0: #first scan for this resonance
    #                     freqFile = open(join(experiment_folder, resonance + "_freq.out"), "w")
    #                     freq = na.query("sens1:x?")
    #                     print "saving frequencies to file"
    #                     freqFile.write(freq)
    #                     freqFile.close()
    #                 print "Waiting for "+str(time_wait[j])+" for data"
    #                 sleep(time_wait[j])
    #                 data = na.query("calc:data? sdata")
    #                 freq_num_str = freq.split(",")
    #                 freq_num = np.array([float(x) for x in freq_num_str])
    #                 data_str = data.split(",")
    #                 dataRaw = np.array([float(x) for x in data_str])
    #                 dataComplex = dataRaw[0::2]+1j*dataRaw[1::2]
    #                 port1 = fitting_tool.my_notch_port()
    #                 port1.add_data(freq_num,dataComplex)
    #
    #                 print pool.apply_async(fit, (port1, os.path.join(experiment_folder,'results',(str(resonance)+'_'+str(power)+'_results.out'))))
    #
    #                 print "saving data to file"
    #                 outFile.write(data+"\n")
    #                 outFile.flush()
    #                 # os.fsync(outFile.fileno()) - causes errors
    #                 outFile.close()
    #                 rIdx=rIdx+1
    #             j=j+1
    #
    # except KeyboardInterrupt:
    #     print "interrupted by Ctrl-C"
    #     ctrlC = True
    # except Exception,e:
    #     traceback.print_exc()
    #     print("\n\n")
    #     print e
    #
    # finally: #clean-up
    #     print "setting power off and trigger hold"
    #     na.sendCommand("OUTP OFF") #set power off at end for safety
    #     na.sendCommand("SENS:SWE:MODE HOLD") #Hold mode
    #
    # if ctrlC:
    #     r = raw_input("Enter Y to delete saved data files...")
    #     if r=="Y":
    #         from shutil import rmtree
    #         rmtree(experiment_folder)
    #
    #
    #
    #
    # na.close()
    # a = raw_input('press enter to close window... \n') #TODO: make a better stopping mechanisem