import sys
from experiment import Experiment, build_tree
sys.path.insert(0,r'../')
import fitting_tool
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import Tkinter as tk
import ttk
# import pwr_sweep_nums5_7_fit


def show_tree(tree, depth=0):
    for branch in tree:
        print '|\t' * depth + '|' + branch
        show_tree(tree[branch], depth + 1)



def main():
    r = 0
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = screen_width / 2 - screen_width * 0.75 / 2
    y = screen_height / 2 - screen_height * 0.75 / 2
    root.geometry(str(int(screen_width*0.75))+'x'+str(int(screen_height*0.75)))
    root.geometry("+%d+%d" % (int(x), int(y)))
    experiment1 = Experiment(freqency_list=['121.375e6', '66.475e6'], powers_list=[20, 30])
    experiment1.initialize()
    variable = tk.StringVar(root)

    w = ttk.Combobox(root,values=experiment1.ordered_supported_parameters.keys(), state="readonly")
    w.grid(row=r, column=0)
    e = ttk.Entry(root)
    e.grid(row=r, column=1, sticky='N S E W')
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=1)
    add = ttk.Button(root)
    add.grid(row=-1, column=-1)
    # root = build_tree(experiment1.Tree)
    # pwr_sweep_nums5_7_fit.run(experiment1, root)
    # experiment1.power_sweep_nums5_7_fit()
    root.mainloop()

if __name__=='__main__':
    main()
