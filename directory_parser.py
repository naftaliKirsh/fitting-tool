import numpy as np
import fitting_tool
import matplotlib.pyplot as plt
import os
import argparse


PATH = r'C:\Users\Owner\Desktop\test data 1'
FREQFILE = '_freq.out'
DATAFILE = '_data.out'
OUTPUT_FOLDER = 'results'


# for file in os.listdir(PATH):
#     if file.endswith(FORMAT):
#         print unichr(10003), file
#     else:
#         print 'X\t', file


def show_info():
    pass


def is_type(file, type):
    if isinstance(file, basestring):
        if file.endswith(type):
            return True
        else:
            return False
    if isinstance(file, list):
        if any(item for item in file if is_type(item, type)):
            return True
        else:
            return False


dirs = [dir for dir in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, dir))]
if not os.path.exists(os.path.join(PATH, OUTPUT_FOLDER)):
    os.mkdir(os.path.join(PATH, OUTPUT_FOLDER))
else:
    print 'ERROR: results alredy exsists delete folder manualy to continue'
    exit(1)
for dir in dirs:
    new_path = os.path.join(PATH, dir)
    result_path = os.path.join(PATH, OUTPUT_FOLDER, dir)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    print '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t directory: ',dir
    files = os.listdir(new_path)
    if is_type(files, FREQFILE):  # TODO: make shure that there is onliy one frequency file in a directory
        for file in files:
            if is_type(file, FREQFILE):
                freq_file = os.path.join(new_path, file)
                data_files = files
                data_files.remove(file)
                for index in range(0,len(data_files)):
                    data_files[index] = os.path.join(new_path,data_files[index])
                for data_file in data_files:
                    print '\t\t\tfile: ',data_file.split('\\')[-1]
                    delay, fit_results = fitting_tool.fit(freq_file, data_file)
                    print delay, fit_results
                    with open(os.path.join(result_path,'results.txt'),'a') as fo:
                        fo.write(str(delay) + ' ' + str(fit_results) + '\n')
                    plt.plot(fitting_tool.port1.f_data * 1e-9, np.absolute(fitting_tool.port1.z_data),
                             label='rawdata', linewidth=0.5)

                    # plt.plot(fitting_tool.port1.f_data * 1e-9, np.absolute(fitting_tool.port1.z_data_sim), label='fit')
                    plt.xlabel('f (GHz)')
                    plt.ylabel('|S21|')
                    leg = plt.legend()
        plt.show()
    else:
        raise Exception('_freq.out FILE NOT FOUND in ' + str(dir) + ' make shure there is such a file in evety directory') #TODO: remove before production


