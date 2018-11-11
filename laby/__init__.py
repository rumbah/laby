import re
import numpy as np
from scipy import odr
from scipy.stats import chi2
import matplotlib.pyplot as plt
from decimal import Decimal
from uncertainties import ufloat, unumpy

def columns(data):
    mat = np.array([[float(x) for x in row.split()] for row in data.strip().splitlines()])
    return mat.transpose()

class Data(odr.RealData):
    def __init__(self, x, y, sx=None, sy=None, x_name=None, y_name=None, x_unit=None, y_unit=None):
        super().__init__(x, y, sx=sx, sy=sy)
        self.x_name = x_name or 'x'
        self.y_name = y_name or 'y'
        self.x_unit = x_unit or ''
        self.y_unit = y_unit or ''

    def fit(self, model, *args, **kwargs):
        return model.fit(self, *args, **kwargs)

    def linear_fit(self, *args, **kwargs):
        return self.fit(LinearModel, *args, **kwargs)

    def quad_fit(self, *args, **kwargs):
        return self.fit(QuadraticModel, *args, **kwargs)

    def log_fit(self, *args, **kwargs):
        return self.fit(LogarithmicModel, *args, **kwargs)

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


    def plot(self, plt=plt, error_fill=False, size=(20, 10), title=None, residuals=True, **kwargs):
        fig, ax = plt.subplots(figsize=size)
        data = self.data
        xmin, xmax, xcnt = data.x.min(), data.x.max(), len(data.x)
        x_size = abs(xmax - xmin)
        x_model = np.linspace(xmin - x_size * 0.1, xmax + x_size * 0.1, xcnt * 3)
        plt.plot(x_model, self.model.fcn(self.beta, x_model), 'black')
        if data.sx is not None:
            plt.errorbar(data.x, data.y, xerr=data.sx, yerr=data.sy, fmt='s', color='b', visible=False, alpha=0.6, ecolor='k',
                capsize=2, capthick=0.5)

        plt.scatter(data.x, data.y, facecolor='red', marker='s',edgecolor='black', s=70, alpha=1)
        plt.rc('font', size=25)
        if title is None:
            title = self._default_title()
        ax.set_title(title)
        x_name = getattr(data, "x_name", 'x')
        if getattr(data, 'x_unit', None):
            x_name = '{} [{}]'.format(x_name, data.x_unit)
        y_name = getattr(data, "y_name", 'y')
        if getattr(data, 'y_unit', None):
            y_name = '{} [{}]'.format(y_name, data.y_unit)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_xlim(x_model.min(), x_model.max())
        ax.grid(True)
        fig.set_facecolor('w')

        if error_fill:
            sigma_ab = np.sqrt(np.diagonal(self.cov_beta))
            bound_upper = self.model.fcn(x_model, *(self.beta + sigma_ab))
            bound_lower = self.model.fcn(x_model, *(self.beta - sigma_ab))
            # plotting the confidence intervals
            plt.fill_between(x_model, bound_lower, bound_upper, color='midnightblue', alpha=0.15)

        if residuals:
            data = self.data
            xmin, xmax, xcnt = data.x.min(), data.x.max(), len(data.x)
            x_size = abs(xmax - xmin)
            x_model = np.linspace(xmin - x_size * 0.1, xmax + x_size * 0.1, xcnt * 3)
            residuals = data.y - self.model.fcn(self.beta, data.x)
            fig, ax = plt.subplots(figsize=size)
            plt.scatter(data.x, residuals, facecolor='red', marker='s', edgecolor='black', s=70, alpha=1)
            ax.errorbar(data.x, residuals, xerr=data.sx, fmt='s', yerr=data.sy, marker='s',
                        visible=False, alpha=0.6, capsize=10, capthick=0.5, ecolor='k')
            ax.set_xlim(x_model.min(), x_model.max())
            ax.set_ylim(residuals.min() - np.abs(residuals.min()*0.5), residuals.max() + np.abs(residuals.max()*0.5))
            ax.set_title(title + " - Residuals Plot")
            resid_y_title = "{} - f({})".format(getattr(data, "y_name", 'y'), getattr(data, 'x_name', 'x'))
            if getattr(data, 'y_unit', None):
                resid_y_title = '{} [{}]'.format(resid_y_title, data.y_unit)
            ax.set_ylabel(resid_y_title)
            ax.set_xlabel(x_name)
            ax.set_xlim(x_model.min(), x_model.max())
            ax.axhline(color='red')
            ax.grid(True)
            fig.set_facecolor('w')

    def _default_title(self):
            return '{} Fit of {}({})'.format(
                getattr(self.model, "name", ''), 
                getattr(self.data, 'y_name', 'y'), 
                getattr(self.data, 'x_name', 'x'))

    def pprint(self):
        print("Fit parameters:")
        for i, a in enumerate(self.params):
            print("  a{} = {:.2u}".format(i+1, a))
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
        if name is not None:
            self.name = name
        self.params = params

    def fit(self, data, guess=None, simple=False):
        if guess is None:
            guess = [0] * self.params
        x = odr.ODR(data, self, beta0=guess)
        x.set_job(fit_type = 2 if simple else 0)
        output = x.run()

        return Output(output, data, self)

LinearModel = Model(lambda a, x: sum(a * [x, 1]), name='Linear', params=2)
QuadraticModel = Model(lambda a, x: sum(a * [x**2, x, 1]), name='Quadratic', params=3)
ExponentialModel = Model(lambda a, x: a[1] * np.e ** (a[0] * x) + a[2], name='Exponential', params=3)
LogarithmicModel = Model(lambda a, x: a[0] * np.log(a[1] + x) + a[2], name='Logarithmic', params=3)
InverseModel = Model(lambda a, x: sum(a * [1 / x, 1]), name='Inverse', params=2)


def test():
    x, y, z = columns("1 2 3\n4 5 6\n7 8 9")
    print(x)
    print(y)
    print(z)

#if __name__ == '__main__':
#    test()