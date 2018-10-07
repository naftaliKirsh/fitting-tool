import experiment
from experiment import Experiment, build_tree
from fitting_tool import SUPPORTED_FIT_RESULTS
from experiment import Ordered_supported_parameters
import Tkinter as tk
import ttk
import tkMessageBox as msgbox
import fake_device as device
import report_generator
import report_generator
import json


global r
r=0

global c
c=0

global variables
variables={}

global plots
plots={}



def main():
    def variables_scrollbar_limits_set(event):
        variables_canvas.configure(scrollregion=variables_canvas.bbox("all"), width=variables_frame.winfo_width(), height=variables_frame.winfo_height())

    def plot_scrollbar_limits_set(event):
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"), width=plot_frame.winfo_width(), height=180)

    def resize_variable_frame(event):
        variables_canvas.itemconfig(in_canvas_variable_frame_id, width=variables_canvas.winfo_width()-variable_scrollbar.winfo_width())

    def resize_plot_frame(event):
        plot_canvas.itemconfig(in_canvas_plot_frame_id, height=plot_canvas.winfo_height()-plot_scrollbar.winfo_height())

    def get_coordinates(event):
        global xpos
        xpos=event.x
        global ypos
        ypos=event.y

    def _on_mousewheel(event):
        if root.winfo_containing(xpos, ypos) != None: #TODO: improove scroll detection
            variables_canvas.yview_scroll(-1 * (event.delta / 120), "units")

    def add_variable(Variable_name='', Varible_value=''):
        def delete_variable():
            variable_name.destroy()
            variable_value.destroy()
            varialve_delete_butotn.destroy()
            variables.pop(variable_name)
        global r
        varialve_delete_butotn = ttk.Button(in_canvas_variables_frame, text='X', command=delete_variable)
        varialve_delete_butotn.grid(row=r)

        variable_name = ttk.Combobox(in_canvas_variables_frame, values=experiment.Ordered_supported_parameters.keys(),
                                     state="readonly")
        if Variable_name != '':
            variable_name.current(Variable_name)
        variable_name.grid(row=r, column=1)

        variable_value = ttk.Entry(in_canvas_variables_frame)
        variable_value.grid(row=r, column=2, sticky='E W')
        variable_value.insert(tk.END, Varible_value)

        variables[variable_name] = variable_value
        r+=1

    def check_plot_validity(box, other_boxes, *args):
        for other_box_variable in other_boxes:
            for list_variable in args:
                if other_box_variable.get() in list_variable:
                    box['values'] = list_variable.keys()
                    return True
        else:
            combined_list = [item for sublist in args for item in sublist.keys()]
            box['values'] = combined_list

    def add_plot(Plot_name=None, X_variable_name='', Y_variable_name=''):
        global c
        if Plot_name == None:
            Plot_name = 'Plot '+str(c+1)
        def delete_plot():
            specific_plot_frame.destroy()
            plots.pop(plot_name)
        specific_plot_frame = ttk.Frame(in_canvas_plot_frame, relief=tk.RIDGE)
        specific_plot_frame.grid(row=0, column=c, sticky='news')
        plot_name = ttk.Entry(specific_plot_frame)
        plot_name.insert(tk.END, Plot_name)
        plot_name.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        plot_lable = ttk.Label(specific_plot_frame, text='x:')
        plot_lable.grid(row=1, column=0, ipady=5, padx=10)
        x_variable_name = ttk.Combobox(specific_plot_frame, values=experiment.Ordered_supported_parameters.keys(),
                                       state="readonly", postcommand=lambda: check_plot_validity(x_variable_name, [y_variable_name], SUPPORTED_FIT_RESULTS, Ordered_supported_parameters))
        x_variable_name.grid(row=1, column=1, padx=10)
        plot_y_lable = ttk.Label(specific_plot_frame, text='y:')
        plot_y_lable.grid(row=2, column=0, pady=10)
        y_variable_name = ttk.Combobox(specific_plot_frame, values=experiment.Ordered_supported_parameters.keys(),
                                       state="readonly", postcommand=lambda: check_plot_validity(y_variable_name, [x_variable_name], SUPPORTED_FIT_RESULTS, Ordered_supported_parameters))
        y_variable_name.grid(row=2, column=1, pady=10)
        spacing = tk.Label(specific_plot_frame)
        spacing.grid(row=3)
        plot_delete_buttion = ttk.Button(specific_plot_frame, text='X', command=delete_plot)
        plot_delete_buttion.grid(row=4, column=0, columnspan=2, sticky='nsew')

        plots[plot_name] = (x_variable_name, y_variable_name)
        c+=1

    def add_3d_plot(Plot_name=None, X_variable_name='', Y_variable_name='', Z_variable_name=''):
        global c
        if Plot_name == None:
            Plot_name = '3d plot '+str(c+1)
        def delete_plot():
            specific_plot_frame.destroy()
            plots.pop(plot_name)
        specific_plot_frame = ttk.Frame(in_canvas_plot_frame, relief=tk.RIDGE)
        specific_plot_frame.grid(row=0, column=c, sticky='news')
        plot_name = ttk.Entry(specific_plot_frame)
        plot_name.insert(tk.END, Plot_name)
        plot_name.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        plot_lable = ttk.Label(specific_plot_frame, text='x:')
        plot_lable.grid(row=1, column=0, ipady=5, padx=10)
        x_variable_name = ttk.Combobox(specific_plot_frame, values=experiment.Ordered_supported_parameters.keys(),
                                       state="readonly", postcommand=lambda: check_plot_validity(x_variable_name, [y_variable_name, z_variable_name], SUPPORTED_FIT_RESULTS, Ordered_supported_parameters))
        x_variable_name.grid(row=1, column=1, padx=10)
        plot_y_lable = ttk.Label(specific_plot_frame, text='y:')
        plot_y_lable.grid(row=2, column=0, pady=10)
        y_variable_name = ttk.Combobox(specific_plot_frame, values=experiment.Ordered_supported_parameters.keys(),
                                       state="readonly", postcommand=lambda: check_plot_validity(y_variable_name, [x_variable_name, z_variable_name], SUPPORTED_FIT_RESULTS, Ordered_supported_parameters))
        y_variable_name.grid(row=2, column=1, pady=10)
        plot_z_lable = ttk.Label(specific_plot_frame, text='z:')
        plot_z_lable.grid(row=3, column=0)
        z_variable_name = ttk.Combobox(specific_plot_frame, values=experiment.Ordered_supported_parameters.keys(),
                                       state="readonly", postcommand=lambda: check_plot_validity(z_variable_name, [y_variable_name, x_variable_name], SUPPORTED_FIT_RESULTS, Ordered_supported_parameters))
        z_variable_name.grid(row=3, column=1)
        plot_delete_buttion = ttk.Button(specific_plot_frame, text='X', command=delete_plot)
        plot_delete_buttion.grid(row=4, column=0, columnspan=2, sticky='nsew')

        plots[plot_name] = (x_variable_name, y_variable_name, z_variable_name)
        c+=1

    def make():
        try:
            vars = []
            parameters={}
            for var in variables.keys():
                vars.append(var.get())
            if len(vars) != len(set(vars)):
                msgbox.showerror('Error', 'choosing the same parameter twice is not allowed\n'
                                          'plese delet one')
                return 1

            for plot in plots:
                for axis in plots[plot]:
                    if not axis.get() in vars+SUPPORTED_FIT_RESULTS.keys():
                        if axis.get() == '':
                            msgbox.showerror('Error', 'enpthy field not allowed in plot')
                            return 1
                        else:
                            msgbox.showerror('Error', 'parameter '+axis.get()+' was not set')
                            return 1


            for variable in  variables.keys():
                parameters[variable.get()]=eval(variables[variable].get())

            experiment1 = Experiment(**parameters)
            print 'initializing...'
            experiment1.initialize()
            print 'building...'
            tree_root = build_tree(experiment1.Tree)
            with open(tree_root+'\\parameters.json', 'w') as fo:
                json.dump(experiment1.parameters, fo)
            print 'build done!'
            count_limit = device.run(experiment1, tree_root, experiment1.parameters['freqency_list'], experiment1.parameters['powers_list'])
            print count_limit
            while True:
                if device.counter.value==count_limit:
                    break
            print '\ndone!'
            data = experiment.load_data('./tree')
            plot_list = []
            for plot in plots:
                if len(plots[plot]) == 3:
                    plot_list.append(([plots[plot][0].get(),plots[plot][1].get(),plots[plot][2].get()],))
                elif len(plots[plot]) == 2:
                    plot_list.append(([plots[plot][0].get(), plots[plot][1].get()],))
            # report_generator.make_report(data, plot_list)
            print 'a'

        except:
            msgbox.showerror('Error', 'plese check the following things:\n'
                                      '* all values are in brackets\n'
                                      '* all values are separated by commas\n'
                                      '* make shure that freqency_list and powers are set\n'
                                      'example:\n'
                                      '\t[1.25, 25, 1.235e+3, 1.74e-6]')


    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = screen_width / 2 - screen_width * 0.75 / 2
    y = screen_height / 2 - screen_height * 0.75 / 2
    root.geometry(str(int(screen_width*0.75))+'x'+str(int(screen_height*0.75)))
    root.geometry("+%d+%d" % (int(x), int(y)))

    buttons_frame = ttk.Frame(root)
    buttons_frame.grid(row=0, sticky='w')

    add_button = ttk.Button(buttons_frame, text='add', command=add_variable)
    add_button.grid(row=0, column=0, sticky='nsew')

    new_plot_buttom = ttk.Button(buttons_frame, text='new plot', command=add_plot)
    new_plot_buttom.grid(row=0, column=1, sticky='nsew')

    new_3d_plot_button = ttk.Button(buttons_frame, text='new 3d plot', command=add_3d_plot)
    new_3d_plot_button.grid(row=0, column=2, sticky='nsew')

    make = ttk.Button(buttons_frame, text='Make', command=make)
    make.grid(row=0, column=3, sticky='nsew')

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
    root.bind('<Motion>', get_coordinates)
    in_canvas_variables_frame.bind_all("<MouseWheel>", _on_mousewheel)

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
    root.bind('<Motion>', get_coordinates)
    root.bind_all("<MouseWheel>", _on_mousewheel)

    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=3)
    root.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)
    in_canvas_variables_frame.columnconfigure(0, weight=0)
    in_canvas_variables_frame.columnconfigure(1, weight=0)
    in_canvas_variables_frame.columnconfigure(2, weight=1)

    add_variable(0,'[5,6]')
    add_variable(1,'[2.2, 3.5, 6.7, 8]')
    add_plot()
    add_3d_plot()


    root.mainloop()

if __name__=='__main__':
    main()
