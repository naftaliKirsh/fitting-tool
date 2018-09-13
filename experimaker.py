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


def main():
    def variables_scrollbar_limits_set(event):
        variables_canvas.configure(scrollregion=variables_canvas.bbox("all"), width=variables_frame.winfo_width(), height=variables_frame.winfo_height())

    def plot_scrollbar_limits_set(event):
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"), width=plot_frame.winfo_width(), height=plot_frame.winfo_height())

    def resize_variable_frame(event):
        variables_canvas.itemconfig(in_canvas_variable_frame_id, height=variables_frame.winfo_height(), width=variables_canvas.winfo_width()-variable_scrollbar.winfo_width())

    def resize_plot_frame(event):
        plot_canvas.itemconfig(in_canvas_plot_frame_id, height=plot_frame.winfo_height()-plot_scrollbar.winfo_height(), width=plot_frame.winfo_width())

    def get_coordinates(event):
        global xpos
        xpos=event.x
        global ypos
        ypos=event.y

    def _on_mousewheel(event):
        if root.winfo_containing(xpos, ypos) != None: #TODO: improove scroll detection
            variables_canvas.yview_scroll(-1 * (event.delta / 120), "units")


    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = screen_width / 2 - screen_width * 0.75 / 2
    y = screen_height / 2 - screen_height * 0.75 / 2
    root.geometry(str(int(screen_width*0.75))+'x'+str(int(screen_height*0.75)))
    root.geometry("+%d+%d" % (int(x), int(y)))
    experiment1 = Experiment(freqency_list=['121.375e6', '66.475e6'], powers_list=[20, 30])
    experiment1.initialize()

    buttons_frame = ttk.Frame(root)
    buttons_frame.grid(row=0, sticky='w')

    add_button = ttk.Button(buttons_frame, text='add')
    add_button.grid(row=0, column=0, sticky='nsew')

    new_plot_buttom = ttk.Button(buttons_frame, text='new plot')
    new_plot_buttom.grid(row=0, column=1, sticky='nsew')

    new_3d_plot_button = ttk.Button(buttons_frame, text='new 3d plot')
    new_3d_plot_button.grid(row=0, column=2, sticky='nsew')

    variables_frame = ttk.Frame(root, relief=tk.GROOVE)
    variables_frame.grid(row=1, sticky='nsew')
    variables_canvas = tk.Canvas(variables_frame)#, highlightthickness=0)
    in_canvas_variables_frame = tk.Frame(variables_canvas)
    variable_scrollbar = ttk.Scrollbar(variables_frame, orient=tk.VERTICAL, command=variables_canvas.yview)
    variables_canvas.configure(yscrollcommand=variable_scrollbar.set)
    variable_scrollbar.pack(side="right", fill="y")
    variables_canvas.pack(side="left", fill=tk.X, expand=True)
    in_canvas_variable_frame_id = variables_canvas.create_window((0, 0), window=in_canvas_variables_frame, anchor='nw')
    in_canvas_variables_frame.bind("<Configure>", variables_scrollbar_limits_set)
    variables_canvas.bind("<Configure>", resize_variable_frame)
    # root.bind('<Motion>', get_coordinates)
    # in_canvas_variables_frame.bind_all("<MouseWheel>", _on_mousewheel)

    delete_butotn = ttk.Button(in_canvas_variables_frame, text='X')
    delete_butotn.grid(row=0)

    w = ttk.Combobox(in_canvas_variables_frame, values=experiment1.ordered_supported_parameters.keys(), state="readonly")
    w.grid(row=0, column=1)

    e = ttk.Entry(in_canvas_variables_frame)
    e.grid(row=0, column=2, sticky='E W')

    plot_frame = ttk.Frame(root, relief=tk.GROOVE)
    plot_frame.grid(row=2, sticky='nsew')
    plot_canvas = tk.Canvas(plot_frame)
    in_canvas_plot_frame = tk.Frame(plot_canvas)
    plot_scrollbar = ttk.Scrollbar(plot_frame, orient=tk.HORIZONTAL, command=plot_canvas.xview)
    plot_canvas.configure(xscrollcommand=plot_scrollbar.set)
    plot_scrollbar.pack(side="bottom", fill=tk.X)
    plot_canvas.pack(side="top", fill=tk.X, expand=True)
    in_canvas_plot_frame_id = plot_canvas.create_window((0, 0), window=in_canvas_plot_frame, anchor='nw')
    in_canvas_plot_frame.bind("<Configure>", plot_scrollbar_limits_set)
    plot_canvas.bind("<Configure>", resize_plot_frame)
    # root.bind('<Motion>', get_coordinates)
    # root.bind_all("<MouseWheel>", _on_mousewheel)

    specific_plot_frame = ttk.Frame(in_canvas_plot_frame, relief=tk.RIDGE)
    specific_plot_frame.grid(column=0, sticky='news')
    plot_name = ttk.Entry(specific_plot_frame, text='Plot 1')
    plot_name.grid(row=0, column=0, columnspan=2)
    plot_name = ttk.Entry(specific_plot_frame, text='Plot 2')
    plot_name.grid(row=0, column=5, columnspan=2)
    plot_x_lable = ttk.Label(specific_plot_frame, text='x:')
    plot_x_lable.grid(row=1, column=0, columnspan=5)
    plot_y_lable =ttk.Label(specific_plot_frame, text='y:')
    plot_y_lable.grid(row=2, column=0)






    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=3)
    root.columnconfigure(0, weight=1)
    in_canvas_variables_frame.columnconfigure(0, weight=0)
    in_canvas_variables_frame.columnconfigure(1, weight=0)
    in_canvas_variables_frame.columnconfigure(2, weight=1)


    # root = build_tree(experiment1.Tree)
    # pwr_sweep_nums5_7_fit.run(experiment1, root)
    # experiment1.power_sweep_nums5_7_fit()
    root.mainloop()

if __name__=='__main__':
    main()
