import sys
from experiment import Experiment, build_tree
sys.path.insert(0,r'x:\QubitElectronics\Scripts\Resonators\fitting tool')
import fitting_tool
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import Tkinter as tk

def show_tree(tree, depth=0):
    for branch in tree:
        print '|\t' * depth + '|' + branch
        show_tree(tree[branch], depth + 1)



def main():
    # root = tk.Tk()
    experiment1 = Experiment(freqency_list=['121.375e6','66.475e6'], powers_list=[20,30])
    experiment1.initialize()
    root = build_tree(experiment1.Tree)

    # experiment1.power_sweep_nums5_7_fit()
    # root.mainloop()

if __name__=='__main__':
    main()
