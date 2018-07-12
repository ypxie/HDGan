try:
    from visdom import Visdom
except:
    print('Better install visdom')
import numpy as np
import random
import scipy.misc
from scipy.misc import imsave
from .local_utils import imshow, writeImg, normalize_img
_port = 43426
print('-'*60)
print('Launch python -m visdom.server -port {} to monitor'.format(_port))
print('-'*60)

#---------------------------------------#
#      Class used for plotting loss     #
#      plot_cls = plot_scalar()         #
#      plot_cls.plot(your_loss)         #
#---------------------------------------#
class plot_scalar(object):
    def __init__(self, name='default', env='main', rate=1, handler=None, port=_port):
        """
        Parameters:
        ----------
        name: str
            name of the plot window.
        env: str
            visdom environment specifier
        rate : int
            rate for refrashing plot.
        handler:  Visdom
            if not specified, will call Visdom().
        port:  int
            plotting port, default=8899

        """
        self.__dict__.update(locals())
        self.values = []
        self.steps = []
        if self.handler is None:
            self.handler = Visdom(port=port)
        self.count = 0

    def plot(self, values, step=None):
        org_type_chk = type(values) is list
        if not org_type_chk:
            values = [values]

        len_val = len(values)
        if step is None:
            step = list(range(self.count, self.count+len_val))

        self.count += len_val
        self.steps.extend(step)
        self.values.extend(values)

        if self.count % self.rate == 0 or org_type_chk:
            self.flush()

    def reset(self):
        self.steps = []
        self.values = []

    def flush(self):
        #print('flush the plot. :)')
        assert type(self.values) is list, 'values have to be list'
        if type(self.values[0]) is not list:
            self.values = [self.values]

        n_lines = len(self.values)
        repeat_steps = [self.steps]*n_lines
        steps = np.array(repeat_steps).transpose()
        values = np.array(self.values).transpose()

        assert not np.isnan(values).any(), 'nan error in loss!!!'
        res = self.handler.line(
            X=steps,
            Y=values,
            win=self.name,
            update='append',
            opts=dict(title=self.name, showlegend=True),
            env=self.env
        )

        if res != self.name:
            self.handler.line(
                X=steps,
                Y=values,
                win=self.name,
                env=self.env,
                opts=dict(title=self.name, showlegend=True)
            )

        self.reset()


def plot_img(X=None, win=None, env=None, plot=None, port=_port):
    if plot is None:
        plot = Visdom(port=port)
    if X.ndim == 2:
        plot.heatmap(X=np.flipud(X), win=win,
                     opts=dict(title=win), env=env)
    elif X.ndim == 3:
        # X is BWC
        norm_img = normalize_img(X)
        plot.image(norm_img.transpose(2, 0, 1), win=win,
                   opts=dict(title=win), env=env)


def save_images(X, save_path=None, save=True, dim_ordering='tf'):
    # X: B*C*H*W or list of X
    if type(X) is list:
        return save_images_list(X, save_path, save, dim_ordering)
    else:
        n_samples = X.shape[0]
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1
        nh, nw = rows, n_samples//rows
        if X.ndim == 4:
            # BCHW -> BHWC
            if dim_ordering == 'tf':
                pass
            else:
                X = X.transpose(0, 2, 3, 1)
            h, w, c = X[0].shape[:3]
            hgap, wgap = int(0.1*h), int(0.1*w)
            img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap, c))
        elif X.ndim == 3:
            h, w = X[0].shape[:2]
            hgap, wgap = int(0.1*h), int(0.1*w)
            img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw - wgap))
        else:
            assert 0, 'you have wrong number of dimension input {}'.format(
                X.ndim)

        for n, x in enumerate(X):
            i = n % nw
            j = n // nw
            rs, cs = j*(h+hgap), i*(w+wgap)
            img[rs:rs + h, cs:cs + w] = x

        if c == 1:
            img = img[:, :, 0]

        if save:
            save_image = ((img + 1) / 2 * 255).astype(np.uint8)
            writeImg(save_image, save_path)

        return img


def save_images_list(X_list, save_path=None, save=True, dim_ordering='tf'):

    # X_list: list of X
    # X: B*C*H*W

    X = X_list[0]
    n_samples = X.shape[0]
    nh = n_samples
    nw = len(X_list)

    if X.ndim == 4:
        # BCHW -> BHWC
        if dim_ordering == 'tf':
            pass
        else:
            for idx, X in enumerate(X_list):
                X_list[idx] = X.transpose(0, 2, 3, 1)

        X = X_list[0]
        h, w, c = X[0].shape[:3]
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap, c))

    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        c = 0
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw - wgap))
    else:
        assert 0, 'you have wrong number of dimension input {}'.format(X.ndim)

    for n, x_tuple in enumerate(zip(*X_list)):
        i = n
        for j, x in enumerate(x_tuple):
            rs, cs = i*(h+hgap), j*(w+wgap)
            img[rs:rs+h, cs:cs+w] = x

    if c == 1:
        img = img[:, :, 0]

    if save:
        save_image = ((img + 1) / 2 * 255).astype(np.uint8)
        writeImg(save_image, save_path)
    return img
