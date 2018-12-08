import itertools
import re
import numpy as np
from scipy import odr
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from decimal import Decimal
from uncertainties import ufloat, UFloat

def columns(data):
    mat = np.array([[float(x) for x in row.split()] for row in data.strip().splitlines()])
    return mat.transpose()

def _get_err(arr, override):
    if len(arr) > 0 and all(hasattr(x, 's') and hasattr(x, 'n') for x in arr):
        return zip(*[(x.n, x.s) for x in arr])
    return arr, override

def take_not(arr, indices):
    return np.take(arr, [x for x in range(len(arr)) if x not in indices])

class Data(odr.RealData):
    def __init__(self, x, y, sx=None, sy=None, x_name=None, y_name=None, x_unit=None, y_unit=None):
        x, sx = _get_err(x, sx)
        y, sy = _get_err(y, sy)
        super().__init__(x, y, sx=sx, sy=sy)
        self.x_name = x_name or 'x'
        self.y_name = y_name or 'y'
        self.x_unit = x_unit or ''
        self.y_unit = y_unit or ''

    def fit(self, model, *args, **kwargs):
        if not isinstance(model, Model):
            model = Model(model)
        return model.fit(self, *args, **kwargs)

    def linear_fit(self, *args, **kwargs):
        return self.fit(LinearModel, *args, **kwargs)

    def quad_fit(self, *args, **kwargs):
        return self.fit(QuadraticModel, *args, **kwargs)

    def log_fit(self, *args, **kwargs):
        return self.fit(LogarithmicModel, *args, **kwargs)

    def select(self, indices):
        return Data(
            np.take(self.x, indices), 
            np.take(self.y, indices), 
            np.take(self.sx, indices) if hasattr(self.sx, '__len__') else self.sx, 
            np.take(self.sy, indices) if hasattr(self.sy, '__len__') else self.sy, 
            self.x_name, self.y_name, self.x_unit, self.y_unit)

def parse_unit(s):
    m = re.match(r'(\S+)\s*\[(.*)\]', s)
    if m:
        return m.groups()
    return s, None

def parse_data(string, y_first=True):
    # check if have title row
    names = [None] * 4
    try:
        firstline = string.split('\n')[0]
        columns(firstline)
    except ValueError:
        names = firstline.split()
        string = string[len(firstline):]

    cols = columns(string)

    sx, sy = None, None
    if len(cols) == 1:
        y, = cols
        y_name, y_unit = parse_unit(names[0])
        x = range(len(cols))
    elif len(cols) == 2:
        y, x = cols
        y_name, y_unit = parse_unit(names[0])
        x_name, x_unit = parse_unit(names[1])
    elif len(cols) == 3:
        y, sy, x = cols
        y_name, y_unit = parse_unit(names[0])
        x_name, x_unit = parse_unit(names[2])
    elif len(cols) == 4:
        y, sy, x, sx = cols
        y_name, y_unit = parse_unit(names[0])
        x_name, x_unit = parse_unit(names[2])
    else:
        raise ValueError("Not sure how to parse %d data columns" % len(cols))
    if not y_first:
        x, sx, y, sy = y, sy, x, sx
        x_name, x_unit, y_name, y_unit = y_name, y_unit, x_name, x_unit
    return Data(x, y, sx=sx, sy=sy, x_name=x_name, y_name=y_name, x_unit=x_unit, y_unit=y_unit)

class Output(odr.Output):
    def __init__(self, orig_output, data, model):
        self.data = data
        self.model = model
        self.__dict__.update(orig_output.__dict__)

        self.chi2_reduced = self.res_var
        self.p_value = chi2.sf(self.sum_square, len(data.x) - len(self.beta))
        self.params = [ufloat(x, y) for x, y in zip(self.beta, self.sd_beta)]


    def _init_ax(self, ax, data, title, xlim=None, ylim=None, resid=False):
        if title is None:
            title = self._default_title() + (' - Residuals Plot' if resid else '')
        if not ax.get_title():
            ax.set_title(title)
        if not ax.get_xlabel():
            x_name = getattr(data, "x_name", 'x')
            if getattr(data, 'x_unit', None):
                x_name = '{} [{}]'.format(x_name, data.x_unit)
            ax.set_xlabel(x_name)
            
        if not ax.get_ylabel():
            y_name = getattr(data, "y_name", 'y')
            if resid:
                y_name = '{} - f({})'.format(y_name, getattr(data, "x_name", 'x'))
            if getattr(data, 'y_unit', None):
                y_name = '{} [{}]'.format(y_name, data.y_unit)
            ax.set_ylabel(y_name)
        ax.grid(True)

        xlim_current, ylim_current = (), ()
        if ax.has_data():
            xlim_current, ylim_current = ax.get_xlim(), ax.get_ylim()
        if xlim is not None:
            xlim = tuple(xlim) + xlim_current
            ax.set_xlim(min(xlim), max(xlim))
        if ylim is not None:
            ylim = tuple(ylim) + ylim_current
            ax.set_ylim(min(ylim), max(ylim))
        
    def plot_data(self, ax=None, title=None, color=None, error_fill=False, size=(16, 8)):
        data = self.data
        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            fig.set_facecolor('w')

        xmin, xmax, xcnt = data.x.min(), data.x.max(), len(data.x)
        x_size = abs(xmax - xmin)
        x_model = np.linspace(xmin - x_size * 0.1, xmax + x_size * 0.1, xcnt * 3)

        self._init_ax(ax, data, title, (x_model.min(), x_model.max()))

        if data.sx is not None:
            ax.errorbar(data.x, data.y, xerr=data.sx, yerr=data.sy, fmt='s', color='b', visible=False, alpha=0.6, ecolor='k')

        x, y = data.x, data.y
        if hasattr(self, 'selected_indices'):
            ax.scatter(take_not(data.x, self.selected_indices),
                       take_not(data.y, self.selected_indices),
                       c='grey', marker='x', s=50, alpha=1)
            x, y = self.selected_data.x, self.selected_data.y
        if len(data.x) < 30:
            ax.scatter(x, y, c=color or 'red', marker='s',edgecolor='black', s=40, alpha=1)
        else:
            ax.plot(x, y, c=color or 'red', linewidth=3)
            # too many points, draw a line
        ax.plot(x_model, self.model.fcn(self.beta, x_model), 'black')
        
        if error_fill:
            sigma_ab = np.sqrt(np.diagonal(self.cov_beta))
            bound_upper = self.model.fcn(x_model, *(self.beta + sigma_ab))
            bound_lower = self.model.fcn(x_model, *(self.beta - sigma_ab))
            # plotting the confidence intervals
            ax.fill_between(x_model, bound_lower, bound_upper, color='midnightblue', alpha=0.15)

    def plot_resid(self, ax=None, title=None, color=None, size=(16, 8)):
        data = self.data

        if ax is None:
            fig, ax = plt.subplots(figsize=size)
            fig.set_facecolor('w')


        xmin, xmax, xcnt = data.x.min(), data.x.max(), len(data.x)
        x_size = abs(xmax - xmin)
        x_model = np.linspace(xmin - x_size * 0.1, xmax + x_size * 0.1, xcnt * 3)
        residuals = data.y - self.model.fcn(self.beta, data.x)

        y_diff = max(abs(residuals.min() - max(abs(residuals.min()*0.5), data.sy.max())),
                     abs(residuals.max() + max(abs(residuals.max()*0.5), data.sy.max())))
        ylim = (y_diff, -y_diff)

        self._init_ax(ax, data, title, (x_model.min(), x_model.max()), ylim, resid=True)

        x, y = data.x, residuals
        if hasattr(self, 'selected_indices'):
            ax.scatter(take_not(data.x, self.selected_indices),
                       take_not(residuals, self.selected_indices),
                       c='grey', marker='x', s=40, alpha=1)
            x, y = np.take(self.data.x, self.selected_indices), np.take(residuals, self.selected_indices)

        ax.scatter(x, y, facecolor=color or 'red', 
            marker='x', s=40, alpha=1)
        ax.errorbar(data.x, residuals, xerr=data.sx, fmt='s', yerr=data.sy, marker='s',
                    visible=False, alpha=0.6, ecolor='k')
        ax.axhline(color='red')

    def plot(self, size=(18, 8), horizontal=True, title=None):
        plt.tight_layout()
        plt.rc('font', size=14, family='sans-serif')
        rows, cols = (1, 2) if horizontal else (2, 1)
        w, h = size
        fig, [ax, rax] = plt.subplots(ncols=cols, nrows=rows, figsize=size)
        fig.set_facecolor('w')
        self.plot_data(ax, title=title)
        self.plot_resid(rax, title=title + '- Residuals' if title else title)

    def _default_title(self):
            return '{} Fit of {}({})'.format(
                getattr(self.model, "name", ''), 
                getattr(self.data, 'y_name', 'y'), 
                getattr(self.data, 'x_name', 'x'))

    def pprint(self):
        print("Fit parameters:")
        for i, a in enumerate(self.params):
            print("  a{} = {:.2u} ({:%})".format(i+1, a, rel_err(a)))
        print("X^2 = {:.5f}".format(self.sum_square))
        cr_diff = np.log2(self.chi2_reduced)
        if -1 < cr_diff < 1:
            cr_well = '=~'
        elif -3 < cr_diff < 3:
            cr_well = '<' if cr_diff < 0 else '>'
        else:
            cr_well = '<<' if cr_diff < 0 else '>>'
        print("X^2_reduced = {:.2f} ({} 1)".format(self.chi2_reduced, cr_well))
        if .25 < self.p_value < .75:
            wellness = "GOOD"
        elif .05 < self.p_value < .95:
            wellness = "OK"
        elif .001 < self.p_value < .999:
            wellness = "BAD"
        else:
            wellness = "WTF"
        print("p_value = {:.2f} ({})".format(self.p_value, wellness))


class Model(odr.Model):
    def __init__(self, *args, name=None, params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or 'Model'
        self.params = params

    def fit(self, data, guess=None, exclude=None, simple=False):
        if guess is None:
            guess = [0] * self.params
        x = odr.ODR(data, self, beta0=guess)
        x.set_job(fit_type = 2 if simple else 0)
        output = Output(x.run(), data, self)
        if exclude:
            good_indices = [i for i in range(len(data.x)) if i not in exclude]
            selected_data = data.select(good_indices)
            x = odr.ODR(selected_data, self, beta0=guess)
            x.set_job(fit_type = 2 if simple else 0)
            orig_output = output
            output = Output(x.run(), selected_data, self)
            output.orig_output = output
            output.selected_data = selected_data
            output.data = data
            output.selected_indices = good_indices
        return output

LinearModel = Model(lambda a, x: sum(a * [x, 1]), name='Linear', params=2)
QuadraticModel = Model(lambda a, x: sum(a * [x**2, x, 1]), name='Quadratic', params=3)
ExponentialModel = Model(lambda a, x: a[1] * np.e ** (a[0] * x) + a[2], name='Exponential', params=3)
LogarithmicModel = Model(lambda a, x: a[0] * np.log(a[1] + x) + a[2], name='Logarithmic', params=3)
InverseModel = Model(lambda a, x: sum(a * [1 / x, 1]), name='Inverse', params=2)


def plot_many(outputs, title=None, size=(20, 10), residuals=False, horizontal=False):
    plt.tight_layout()
    plt.rc('font', size=14, family='sans-serif')
    if residuals:
        rows, cols = (1, 2) if horizontal else (2, 1)
    else:
        rows, cols = (1, 1)
    w, h = size
    fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size)
    fig.set_facecolor('w')
    if residuals:
        ax, rax = ax
    colors = itertools.cycle(['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'pink', 'brown'])
    for color, output in zip(colors, outputs):
        if residuals:
            output.plot_resid(rax, color=color, title=title + '- Residuals' if title else title)
        output.plot_data(ax, color=color, title=title)

def Nsigma(measured, expected):
    sdiff = (measured.s ** 2 + expected.s ** 2) ** .5
    return abs(measured.n - expected.n) / sdiff if sdiff else float('inf')

def rel_err(value):
    return abs(value.s / value.n) if value.n else float('inf')

def test():
    x, y, z = columns("1 2 3\n4 5 6\n7 8 9")
    print(x)
    print(y)
    print(z)

#if __name__ == '__main__':
#    test()