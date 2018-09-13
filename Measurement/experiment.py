import sys
from collections import defaultdict, OrderedDict
sys.path.insert(0,r'../')
import fitting_tool
import os


def tree(): return defaultdict(tree)


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
    fit_results = fitting_tool.fit(port1)
    experiment.populate([root]+path, fit_results, '_results.out')
    print fit_results
    fitting_tool.myplotnorm(port1)


class Experiment():
    def __init__(self, ordered_supported_parameters=OrderedDict([('freqency_list', 'Ghz'), ('powers_list', 'db'), ('bios', 'V')]), **kwargs):
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

    def populate(self,path, data, type):
        if not '.' in type:
            type = '.'+type
        path.append(path[-1]+type)
        if not os.path.exists(os.path.join(*path)):
            file = os.path.join(*path)
            with open(file, 'w') as fo:
                fo.write(data)

    def make_fit(self, pool, f_data, complex_data, root, parameters):
        port1 = fitting_tool.my_notch_port()
        port1.add_data(f_data, complex_data)
        self.populate([root, parameters[0]], f_data, '_freq.out')
        self.populate([root]+parameters, complex_data, '_data.out')
        pool.apply_async(fit, (port1, self, root, parameters))
