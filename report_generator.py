import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
from fitting_tool import SUPPORTED_FIT_RESULTS
#TODO: export all package variables gto a seperate file
from matplotlib.backends.backend_pdf import PdfPages


def pair(dictionaries):
    key_list = []
    for dictionary in dictionaries:
        dict_list = []
        for key in dictionary:
            dict_list.append(key)
        key_list.append(dict_list)
    unfiltered_key_combitaitons = list(itertools.product(*key_list))
    #TODO: check common nodes until the lowest one!!!
    data_list = []
    for key_group in unfiltered_key_combitaitons:
        if '\\' in max(key_group, key=len) or '/' in max(key_group, key=len):
            longest_key = os.path.split(max(key_group, key=len))[0]
        else:
            longest_key = max(key_group, key=len)
        modified_key_group = []
        for key in key_group:
            if '\\' in key or '/' in key:
                modified_key_group.append(os.path.split(key)[0])
            else:
                modified_key_group.append(key)
        if all([key in longest_key for key in modified_key_group]):
            spesific_data_list = []
            for i in range(len(key_group)):
                key = key_group[i]
                spesific_data_list.append(dictionaries[i][key])
            data_list.append(spesific_data_list)
    return data_list

def sync(group_list):
    synced_list = []
    for group in group_list:
        if all([isinstance(member, dict) for member in group]):
            lists = []
            keys = group[0].keys()
            for dictionary in group:
                lists.append([dictionary[key] for key in keys])
            synced_list.append(lists)
        else:
            synced_list.append(group)
    return synced_list

def data_array_maker(data, figures):
    for figure in figures:
        for plot in figures:
            for axis in plot:
                if axis in SUPPORTED_FIT_RESULTS.keys():
                    pass
                elif axis in data['parameters'].keys():
                    pass
                else:
                    pass

def make_report(data, plot_list, overlay=False):
    for plot in plot_list:
        specific_plot_dictionaries = []
        for subplot in plot:
            for axis in subplot:
                specific_plot_dictionaries.append(data[axis])
            paired_data = pair(specific_plot_dictionaries)
            for fig in paired_data:
                if not overlay:
                    plt.figure()
                for graphic in plot:
                    if len(graphic) == 2:
                        plt.plot(graphic[0], graphic[1])
                    elif len(graphic) == 3:
                        X, Y = np.meshgrid(graphic[0], graphic[1])
                        fig, ax = plt.subplots()
                        cs = ax.contourf(X, Y, graphic[2])
                        cbar = fig.colorbar(cs)
                    else:
                        raise ValueError('only 2d and 3d plots supported')
            else:
                raise ValueError('all plots must be inside tuples')
        plt.show()


# x=np.linspace(-5,5)
# y=x
# X,Y = np.meshgrid(x, y)
# z=X+Y**2
#
# make_report(tuple([[x,y]]),([x,y**2],[x,x**3]),tuple([[x,y,z]]))