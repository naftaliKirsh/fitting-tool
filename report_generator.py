import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages




def make_report(*args):
    for arg in args:
        if isinstance(arg, tuple):
            plt.figure()
            for graphic in arg:
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


x=np.linspace(-5,5)
y=x
X,Y = np.meshgrid(x, y)
z=X+Y**2

make_report(tuple([[x,y]]),([x,y**2],[x,x**3]),tuple([[x,y,z]]))