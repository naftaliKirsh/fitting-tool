import sys
from collections import defaultdict, OrderedDict
sys.path.insert(0,r'../')
import fitting_tool
import numpy as np
import json
import os

def tree(): return defaultdict(tree)


def load_data(tree_root_path):
    with open(os.path.join(tree_root_path, 'parameters.json'), 'r') as fo:
        parameters = json.load(fo, object_pairs_hook=OrderedDict)
    key = ['parameters']+parameters.keys()+['result files']
    data = {}
    file_list = [os.path.join(root, file) for root, folders, files in os.walk(tree_root_path) for file in files]
    files = [file.replace(os.path.commonprefix(file_list), '') for file in file_list]
    for file in files:
        corrected_parameter_path = len(file.split('\\'))
        if corrected_parameter_path == 1:
            continue
        if not key[corrected_parameter_path-1] in data.keys():
            data[key[corrected_parameter_path - 1]] = {}
        try:
            data[key[corrected_parameter_path - 1]][file] = np.loadtxt(os.path.join(tree_root_path, file))
        except ValueError:
            try:
                data[key[corrected_parameter_path - 1]][file] = np.loadtxt(os.path.join(tree_root_path, file), dtype=complex,
                                                    converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
            except ValueError:
                with open(os.path.join(tree_root_path, file), 'r') as fo:
                    data[key[corrected_parameter_path - 1]][file] = json.load(fo)
    for result_path in data['result files']:
        with open(os.path.join(tree_root_path, result_path), 'r') as fo:
            result = json.load(fo)
        for key in result.keys():
            if not key in data.keys():
                data[key] = {}
            if not (result_path.split('\\')[0]) in data[key].keys():
                data[key][(result_path.split('\\')[0])] = {}
            if not result_path in data[key][(result_path.split('\\')[0])].keys():
                data[key][(result_path.split('\\')[0])][result_path] = {}
            data[key][(result_path.split('\\')[0])][result_path] = result[key]
    for parameter in parameters:
        data['user_input_'+parameter] = {}
        for parameter_path in data[parameter]:
            corrected_parameter_path = parameter_path.split('\\')[0:-1]+['paraeter.name']
            data['user_input_'+parameter][os.path.join(*corrected_parameter_path)] = float(parameter_path.split('\\')[-2])
    return data





def show_tree(tree, depth=0):
    for branch in tree:
        print '|\t' * depth + '|' + branch
        show_tree(tree[branch], depth + 1)

def build_tree(tree, depth=0, root=r'.\tree'):
    if not os.path.exists(root):
        os.mkdir(root)
    for branch in tree:
        if not os.path.exists(os.path.join(root,branch)):
            os.mkdir(os.path.join(root, branch))

        # print '|\t' * depth + '|' + branch
        build_tree(tree[branch], depth + 1, os.path.join(root,branch))
    return root

def add(t, path):
    for node in path:
        t = t[node]

def remove(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def fit(port1, experiment, root, path):
    global counter

    print '\t\t\t\tstart'
    fit_results = fitting_tool.fit(port1)
    print fit_results
    folder = [root]+path+['results']
    if not os.path.exists(os.path.join(*folder)):
        os.mkdir(os.path.join(*folder))
    experiment.populate([root]+path+['results'], fit_results, '.json', format='json')
    with counter.get_lock():
        counter.value += 1
    print counter.value
    fitting_tool.myplotnorm(port1)

def init_process(args):
    global counter
    counter = args

Ordered_supported_parameters = OrderedDict([('freqency_list', 'Ghz'), ('powers_list', 'db'), ('bios', 'V')])


class Experiment():
    def __init__(self, ordered_supported_parameters=Ordered_supported_parameters, **kwargs):
        self.ordered_supported_parameters = ordered_supported_parameters
        if not 'freqency_list' in kwargs.keys():
            raise ValueError('freqency_list not set')
        if not 'powers_list' in kwargs.keys():
            raise ValueError('powers_list not set')
        for parameter in kwargs.keys():
            if parameter not in self.ordered_supported_parameters.keys():
                raise ValueError(str(parameter) + ' is not a supported parameter')
        self.Tree = tree()
        self.parameters = OrderedDict()
        for key in self.ordered_supported_parameters.keys():
            if key in kwargs:
                self.parameters[key] = kwargs[key]

    def initialize(self, tree=None, parameters=None):
        if tree == None:
            tree = self.Tree
        if parameters == None:
            parameters = self.parameters
        if len(parameters) > 0:
            if len(tree) == 0:
                for value in parameters[parameters.keys()[0]]:
                    # add(tree, [str(value)+str(self.ordered_supported_parameters[parameters.keys()[0]])]) #with units
                    add(tree, [str(value)]) #without units
                self.initialize(tree, parameters=remove(parameters, parameters.keys()[0]))
            else:
                for branch in tree:
                    for value in parameters[parameters.keys()[0]]:
                        # add(tree, [str(branch), str(value)+str(self.ordered_supported_parameters[parameters.keys()[0]])])  #with units
                        add(tree, [str(branch), str(value)]) #without units
                    self.initialize(tree[branch], remove(parameters, parameters.keys()[0]))

    def populate(self,path, data, type, format='numpy'):
        if not '.' in type:
            type = '.'+type
        path.append(path[-1]+type)
        if not os.path.exists(os.path.join(*path)):
            file = os.path.join(*path)
            if format=='string':
                with open(file, 'w') as fo:
                    fo.write(str(data))
            elif format=='json':
                with open(file, 'w') as fo:
                    json.dump(data, fo)
            elif format == 'numpy':
                np.savetxt(file, data)

        # else:
        #     raise IOError('path not found')

    def make_fit(self, pool, f_data, complex_data, root, parameters, Counter):
        global counter
        counter = Counter
        port1 = fitting_tool.my_notch_port()
        port1.add_data(f_data, complex_data)
        self.populate([root, parameters[0]], list(f_data), '_freq.out', format='numpy')
        self.populate([root]+parameters, list(complex_data), '_data.out', format='numpy')
        pool.apply_async(fit, (port1, self, root, parameters))
