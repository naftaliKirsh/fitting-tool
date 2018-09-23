from matplotlib.widgets import Slider, Button
import Tkinter as tk
import tkFileDialog
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import fitting_tool

Ql = 5e4
Qc = 2e5
phi = 0 * np.pi
fr = 5e9
f = np.arange(4.999e9, 5.001e9, 1e2)
a = 1
alpha = 1
delay = 0.0e0
noise_amplitude = 3e5
gain = 1e-7


def update(val):
    global Ql
    Ql = sQl.val
    global Qc
    Qc = sQc.val
    global phi
    phi = sphi.val
    global a
    a = sa.val
    global alpha
    alpha = salpha.val
    global delay
    delay = np.float64(sdelay.val)
    gain = sgain.val
    global S_21
    S_21 = a * np.exp(1j * alpha) * np.exp(-2 * np.pi * 1j * f * delay) * (
                1 - ((Ql / np.abs(Qc) * np.exp(1j * phi)) / (1 + 2j * Ql * (f / fr - 1))))
    S_21 = S_21 + ((noiseRe + 1j * noiseIm) * gain)
    real = S_21.real
    imag = S_21.imag
    l.set_xdata(real)
    l.set_ydata(imag)
    l2.set_ydata(np.absolute(abs(S_21)))
    l3.set_ydata(np.angle(S_21))
    matplotlib.axes.Axes.relim(ax112)
    matplotlib.axes.Axes.autoscale_view(ax112)
    matplotlib.axes.Axes.relim(ax222)
    matplotlib.axes.Axes.autoscale_view(ax222)
    matplotlib.axes.Axes.relim(ax223)
    matplotlib.axes.Axes.autoscale_view(ax223)
    fig.canvas.draw_idle()


def save_data(argument):
    S_21 = a * np.exp(1j * alpha) * np.exp(-2 * np.pi * 1j * f * delay) * (
            1 - ((Ql / np.abs(Qc) * np.exp(1j * phi)) / (1 + 2j * Ql * (f / fr - 1))))
    S_21 = S_21 + ((noiseRe + 1j * noiseIm) * gain)
    root = tk.Tk()
    root.withdraw()
    root.filename = tkFileDialog.asksaveasfilename(initialdir=".", title="Select location and enter file name")
    root.deiconify()
    root.destroy()
    np.savetxt(root.filename + '_freq.out', f, delimiter=',', newline=',')
    with open(root.filename + '_freq.out', 'r+') as fo:
        fo.seek(-1, 2)
        fo.truncate()
    out = zip(S_21.real, S_21.imag)
    np.savetxt(root.filename + '_data.out', out, delimiter=',', newline=',')
    with open(root.filename + '_data.out', 'r+') as fo:
        fo.seek(-1, 2)
        fo.truncate()
    # with open(root.filename+'_freq.out','w') as fo:
    #     for i in f:
    #         fo.write(str(i)+',')
    # with open(root.filename+'_data.out','w') as fo:
    #     out = zip(S_21.real, S_21.imag)
    #     out = str(out).replace('(','')
    #     out = str(out).replace(')','')
    #     out = str(out).replace('[','')
    #     out = str(out).replace(']','')
    #     out = str(out).replace(' ','')
    #     fo.write(out)
    with open(root.filename + '_params.out', 'w') as fo:
        values = [Ql, Qc, phi, fr, a, alpha, delay, noise_amplitude, gain]
        names = ['Ql', 'Qc', 'phi', 'fr', 'a', 'alpha', 'delay', 'noise_amplitude', 'gain']
        out = zip(names, values)
        fo.write(str(out))
    return root.filename + '_freq.out', root.filename + '_data.out'
#TODO: fix gain value in params file

def fit_data(argument):
    FreqFile, DataFile = save_data(argument)
    fit__res = fitting_tool.fit(FreqFile, DataFile, True)
    # print '\t\t\t\t\t\'deley\':   \t\t\t\'fr\':   \t\t\t\'Ql\': '
    # print 'fitted:\t\t\t',
    # for val in fit__res:
    #     print '{:e}'.format(val), '\t\t',
    # print '\ngenerated:\t\t',
    # print '{:e}'.format(delay) + ' \t\t' + '{:e}'.format(fr) + '\t\t' + '{:e}'.format(Ql)  # +', \t\t\
    # # t\'Qc\': '+str(Qc)+',\'phi\': '+str(phi)
    # print '------------------------------------------------------------------------'
    print fit__res
    print delay
    plt.ioff()


if __name__ == '__main__':
    fig, ax = plt.subplots()

    S_21 = a * np.exp(1j * alpha) * np.exp(-2 * np.pi * 1j * f * delay) * (
                1 - ((Ql / np.abs(Qc) * np.exp(1j * phi)) / (1 + 2j * Ql * (f / fr - 1))))
    noiseRe = np.random.normal(0, noise_amplitude, len(S_21))
    noiseIm = 1j * np.random.normal(0, noise_amplitude, len(S_21))
    S_21 = S_21 + (noiseRe + 1j * noiseIm) * gain

    real = S_21.real
    imag = S_21.imag
    ax112 = plt.subplot(221)
    l, = plt.plot(real, imag)
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    ax222 = plt.subplot(222)
    l2, = plt.plot(f, np.absolute(abs(S_21)))
    plt.xlabel('f (GHz)')
    plt.ylabel('|S21|')
    ax223 = plt.subplot(223)
    l3, = plt.plot(f, np.angle(S_21))
    plt.xlabel('f (GHz)')
    plt.ylabel('arg(|S21|)')

    axcolor = 'lightgoldenrodyellow'
    axphi = plt.axes([0.55, 0.1, 0.35, 0.03], facecolor=axcolor)
    axQl = plt.axes([0.55, 0.15, 0.35, 0.03], facecolor=axcolor)
    axQc = plt.axes([0.55, 0.20, 0.35, 0.03], facecolor=axcolor)
    axa = plt.axes([0.55, 0.25, 0.35, 0.03], facecolor=axcolor)
    axalpha = plt.axes([0.55, 0.30, 0.35, 0.03], facecolor=axcolor)
    axdelay = plt.axes([0.55, 0.35, 0.35, 0.03], facecolor=axcolor)
    axgain = plt.axes([0.55, 0.40, 0.35, 0.03], facecolor=axcolor)

    sphi = Slider(axphi, 'phi', 0, 2 * np.pi, valinit=phi, valfmt='%.2e')
    sQl = Slider(axQl, 'Ql', 0, 2 * Ql, valinit=Ql, valfmt='%.2e')
    sQc = Slider(axQc, 'Qc', 0, 2 * Qc, valinit=Qc, valfmt='%.2e')
    sa = Slider(axa, 'a', 0.01, 100, valinit=a)
    salpha = Slider(axalpha, 'alpha', 0, 2 * alpha, valinit=alpha, valfmt='%.2e')
    sdelay = Slider(axdelay, 'delay', 0, 1e-6, valinit=delay, valfmt='%.2e')
    sgain = Slider(axgain, 'gain', 0, 2 * gain, valinit=gain, valfmt='%.2e')

    sphi.on_changed(update)
    sQl.on_changed(update)
    sQc.on_changed(update)
    sa.on_changed(update)
    salpha.on_changed(update)
    sdelay.on_changed(update)
    sgain.on_changed(update)

    axsave = plt.axes([0.9, 0.0, 0.1, 0.075])
    save = Button(axsave, 'save', hovercolor='green')
    save.on_clicked(save_data)
    axcut = plt.axes([0.8, 0.0, 0.1, 0.075])
    fit = Button(axcut, 'fit', hovercolor='green')
    fit.on_clicked(fit_data)

    plt.show()
