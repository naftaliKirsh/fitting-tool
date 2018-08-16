from resonator_tools import circuit, circlefit
import numpy as np
import Tkinter, tkFileDialog
import matplotlib.pyplot as plt
import scipy.optimize as spopt
from os.path import join
import os
import pprint
from matplotlib import pyplot

global Scale
Scale = 100

global debug
debug = False


def file_chooser(format):
    root = Tkinter.Tk()
    root.withdraw()  # hide root
    root.File = tkFileDialog.askopenfilename(filetypes=(("data files", format), ("all files", "*.*")))
    root.deiconify()
    root.destroy()
    return root.File


def MyGUIfit(self):
    '''
    automatic fit with possible user interaction to crop the data and modify the electric delay
    f1,f2,delay are determined in the GUI. Then, data is cropped and autofit with delay is performed
    '''
    # copy data
    fmin, fmax = self.f_data.min(), self.f_data.max()
    myfit(self)
    self.__delay = self._delay
    # prepare plot and slider
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    fig, ((ax2, ax0), (ax1, ax3)) = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Normalized data. Use the silders to improve the fitting if necessary.')
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l0, = ax0.plot(self.f_data * 1e-9, np.absolute(self.z_data) - (np.absolute(self.z_data_sim_norm[self._fid])))
    l1, = ax1.plot(self.f_data * 1e-9, np.angle(self.z_data))
    l2, = ax2.plot(np.real(self.z_data) - (np.real(self.z_data_sim_norm[self._fid])),
                   np.imag(self.z_data) - (np.imag(self.z_data_sim_norm[self._fid])))
    # l0s, = ax0.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim_norm))
    # l1s, = ax1.plot(self.f_data * 1e-9, np.angle(self.z_data_sim_norm))
    # l2s, = ax2.plot(np.real(self.z_data_sim_norm), np.imag(self.z_data_sim_norm))
    ax0.set_xlabel('f (GHz)')
    ax1.set_xlabel('f (GHz)')
    ax2.set_xlabel('real')
    ax0.set_ylabel('amp')
    ax1.set_ylabel('phase (rad)')
    ax2.set_ylabel('imagl')
    fr_ann = ax3.annotate('fr = %e Hz +- %e Hz' % (self.fitresults['fr'], self.fitresults['fr_err']), xy=(0.1, 0.8),
                          xycoords='axes fraction')
    Ql_ann = ax3.annotate('Ql = %e +- %e' % (self.fitresults['Ql'], self.fitresults['Ql_err']), xy=(0.1, 0.6),
                          xycoords='axes fraction')
    Qc_ann = ax3.annotate('Qc = %e +- %e' % (self.fitresults['absQc'], self.fitresults['absQc_err']), xy=(0.1, 0.4),
                          xycoords='axes fraction')
    Qi_ann = ax3.annotate('Qi = %e +- %e' % (self.fitresults['Qi_dia_corr'], self.fitresults['Qi_dia_corr_err']),
                          xy=(0.1, 0.2), xycoords='axes fraction')
    axcolor = 'lightgoldenrodyellow'
    axdelay = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
    axf2 = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axf1 = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
    sscale = 10.
    sdelay = Slider(axdelay, 'delay', -1., 1., valinit=self.__delay / (sscale * self.__delay), valfmt='%f')
    df = (fmax - fmin) * 0.05
    sf2 = Slider(axf2, 'f2', (fmin - df) * 1e-9, (fmax + df) * 1e-9, valinit=fmax * 1e-9, valfmt='%.10f GHz')
    sf1 = Slider(axf1, 'f1', (fmin - df) * 1e-9, (fmax + df) * 1e-9, valinit=fmin * 1e-9, valfmt='%.10f GHz')

    def update(val):
        self.autofit(electric_delay=sdelay.val * sscale * self.__delay, fcrop=(sf1.val * 1e9, sf2.val * 1e9))
        l0.set_data(self.f_data * 1e-9, np.absolute(self.z_data) - (np.absolute(self.z_data_sim_norm[self._fid])))
        l1.set_data(self.f_data * 1e-9, np.angle(self.z_data) - (np.angle(self.z_data_sim_norm[self._fid])))
        l2.set_data(np.real(self.z_data) - (np.real(self.z_data_sim_norm[self._fid])),
                    np.imag(self.z_data) - (np.imag(self.z_data_sim_norm[self._fid])))

        print 'abs=', max(np.absolute(self.z_data) - (np.absolute(self.z_data_sim_norm[self._fid]))) * Scale
        print 'ang=', max(np.angle(self.z_data) - (np.angle(self.z_data_sim_norm[self._fid]))) * Scale
        print 'Re=', max(np.real(self.z_data) - (np.real(self.z_data_sim_norm[self._fid]))) * Scale
        print 'Im=', max(np.imag(self.z_data) - (np.imag(self.z_data_sim_norm[self._fid]))) * Scale, '\n'

        # l0s.set_data(self.f_data[self._fid] * 1e-9, np.absolute(self.z_data_sim_norm[self._fid]))
        # l1s.set_data(self.f_data[self._fid] * 1e-9, np.angle(self.z_data_sim_norm[self._fid]))
        # l2s.set_data(np.real(self.z_data_sim_norm[self._fid]), np.imag(self.z_data_sim_norm[self._fid]))
        fr_ann.set_text('fr = %e Hz +- %e Hz' % (self.fitresults['fr'], self.fitresults['fr_err']))
        Ql_ann.set_text('Ql = %e +- %e' % (self.fitresults['Ql'], self.fitresults['Ql_err']))
        Qc_ann.set_text('|Qc| = %e +- %e' % (self.fitresults['absQc'], self.fitresults['absQc_err']))
        Qi_ann.set_text('Qi_dia_corr = %e +- %e' % (self.fitresults['Qi_dia_corr'], self.fitresults['Qi_dia_corr_err']))
        fig.canvas.draw_idle()

    def btnclicked(event):
        self.autofit(electric_delay=None, fcrop=(sf1.val * 1e9, sf2.val * 1e9))
        self.__delay = self._delay
        sdelay.reset()
        update(event)

    sf1.on_changed(update)
    sf2.on_changed(update)
    sdelay.on_changed(update)
    btnax = plt.axes([0.05, 0.1, 0.1, 0.04])
    button = Button(btnax, 'auto-delay', color=axcolor, hovercolor='0.975')
    button.on_clicked(btnclicked)
    plt.show()
    plt.close()


def myplot1(self, i):
    plt.figure(i)
    real = self.z_data_raw.real
    imag = self.z_data_raw.imag
    real2 = self.z_data_sim.real
    imag2 = self.z_data_sim.imag
    plt.subplot(221)
    plt.plot(real - real2, imag - imag2, label='rawdata')
    # plt.plot(real, imag, label='rawdata')
    # plt.plot(real2, imag2, label='fit')
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    # plt.legend()
    plt.subplot(222)
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw) - (np.absolute(self.z_data_sim)), label='rawdata')
    # plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label='rawdata')
    # plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('|S21|')
    # plt.legend()
    plt.subplot(223)
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw) - (np.angle(self.z_data_sim)), label='rawdata')
    # plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label='rawdata')
    # plt.plot(self.f_data * 1e-9, np.angle(self.z_data_sim), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('arg(|S21|)')
    # plt.legend()
    # plt.show()


def myplot2(self, i):
    plt.figure(i)
    real = self.z_data_raw.real
    imag = self.z_data_raw.imag
    real2 = self.z_data_sim.real
    imag2 = self.z_data_sim.imag
    plt.subplot(221)
    plt.plot(real, imag, label='rawdata')
    plt.plot(real2, imag2, label='fit')
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    plt.legend()
    plt.subplot(222)
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label='rawdata')
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('|S21|')
    plt.legend()
    plt.subplot(223)
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label='rawdata')
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data_sim), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('arg(|S21|)')
    plt.legend()
    # plt.show()


def myplotnorm(self):
    real = np.real(self.z_data)
    imag = np.imag(self.z_data)
    real2 = np.real(self.z_data_sim_norm)
    imag2 = np.imag(self.z_data_sim_norm)
    plt.subplot(221)
    plt.plot(real, imag, label='rawdata')
    plt.plot(real2, imag2, label='fit')
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    plt.legend()
    plt.subplot(222)
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data), label='rawdata')
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim_norm), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('|S21|')
    plt.legend()
    plt.subplot(223)
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data), label='rawdata')
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data_sim_norm), label='fit')
    plt.xlabel('f (GHz)')
    plt.ylabel('arg(|S21|)')
    plt.legend()
    plt.show()


def myfit(self, electric_delay=None, fcrop=None):
    '''
    automatic calibration and fitting
    electric_delay: set the electric delay manually
    fcrop = (f1,f2) : crop the frequency range used for fitting
    '''
    if fcrop is None:
        self._fid = np.ones(self.f_data.size, dtype=bool)
    else:
        f1, f2 = fcrop
        self._fid = np.logical_and(self.f_data >= f1, self.f_data <= f2)
    delay, amp_norm, alpha, fr, Ql, A2, frcal = \
        self.do_calibration(self.f_data[self._fid], self.z_data_raw[self._fid], ignoreslope=True, guessdelay=True,
                            fixed_delay=electric_delay)
    self.z_data = self.do_normalization(self.f_data, self.z_data_raw, delay, amp_norm, alpha, A2, frcal)
    self.fitresults = self.circlefit(self.f_data[self._fid], self.z_data[self._fid], fr, Ql, refine_results=False,
                                     calc_errors=True)
    self.z_data_sim = A2 * (self.f_data - frcal) + self._S21_notch(self.f_data, fr=self.fitresults["fr"],
                                                                   Ql=self.fitresults["Ql"],
                                                                   Qc=self.fitresults["absQc"],
                                                                   phi=self.fitresults["phi0"], a=amp_norm, alpha=alpha,
                                                                   delay=delay)
    self.z_data_sim_norm = self._S21_notch(self.f_data, fr=self.fitresults["fr"], Ql=self.fitresults["Ql"],
                                           Qc=self.fitresults["absQc"], phi=self.fitresults["phi0"], a=1.0, alpha=0.,
                                           delay=0.)
    self._delay = delay
    try:
        self._errors = [port1.fitresults['fr_err'] * 1e-3, port1.fitresults[
            'Ql_err'] * 1e-3]  # TODO: fix WARNING: Error calculation failed! originating in circuir.py line: 410 due to matrix A being singular
    except KeyError:
        self._errors = [None, None]
    self._tests = [max(np.absolute(self.z_data) - (np.absolute(self.z_data_sim_norm[self._fid]))) * 100,
                   max(np.angle(port1.z_data) - (np.angle(port1.z_data_sim_norm[port1._fid]))) * 100,
                   max(np.real(port1.z_data) - (np.real(port1.z_data_sim_norm[port1._fid]))) * 100,
                   max(np.imag(port1.z_data) - (np.imag(port1.z_data_sim_norm[port1._fid]))) * 100]


def search_dddelay(delay, radius=5e-8,
                   resolution=3.5e-9,
                   verbose=False):  # TODO: error when resolution is too small with relation to radius
    diffs = {}
    for d in np.arange(delay - radius, delay + radius, resolution):
        myfit(port1, d)
        diffs[d] = (port1._tests)
    if verbose:
        plt.plot(diffs.keys(), diffs.values(), '.')
        plt.show(block=False)
    sd = []
    for x in diffs.values():
        sd.append(sum(map(abs, x)))
    dddelay = diffs.keys()[sd.index(min(sd))]
    return dddelay


def smart_search_delay(start_value, depth, verbose=False):
    delay = start_value
    radius = 1e-7
    resolution = 1e-8
    for level in range(1, depth + 1):
        delay = search_dddelay(delay, radius, resolution, verbose)
        radius = radius * 1e-1
        resolution = resolution * 1e-1
    return delay


def fit(FreqFile, DataFile, verbose=False):
    try:
        freq = np.loadtxt(FreqFile, delimiter=',')
        dataRaw = np.loadtxt(DataFile, delimiter=',')
    except IOError:
        print 'ERROR: file not found'
        exit(1)
    data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]

    global port1
    port1 = circuit.notch_port()
    port1.add_data(freq, data)
    myfit(port1)
    dddelay = smart_search_delay(port1._delay, 3, verbose=verbose)
    port1.autofit(dddelay)
    return dddelay, port1.fitresults


if __name__ == '__main__':
    if debug:
        FreqFile = r'.\15_03_18 cooldown\Yaakov (channel 5)\PowerScan\4.71865e9_freq.out'
        DataFile = r'.\15_03_18 cooldown\Yaakov (channel 5)\PowerScan\4.71865e9_-45.0_data.out'
    else:
        FreqFile = file_chooser('*_freq.out')
        DataFile = file_chooser('*_data.out')

    try:
        freq = np.loadtxt(FreqFile, delimiter=',')
        dataRaw = np.loadtxt(DataFile, delimiter=',')
    except IOError:
        print 'ERROR: file not found'
        exit(1)
    data = dataRaw[0:-1:2] + 1j * dataRaw[1::2]

    port1 = circuit.notch_port()
    port1.add_data(freq, data)
    myfit(port1)
    dddelay = smart_search_delay(port1._delay, 3, verbose=True)  # TODO: expose depth to user via gui
    port1.autofit(dddelay)
    print dddelay, port1.fitresults
    plt.figure(2)
    plt.ion()
    myplotnorm(port1)
    plt.figure(2)
    plt.ioff()
    port1.autofit()
    print port1._delay, '      ', port1.fitresults['fr']
    port1.GUIfit()
