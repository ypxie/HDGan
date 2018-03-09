try:
    from visdom import Visdom
except:
    print('Better install visdom')
import numpy as np
import random

import scipy.misc
from scipy.misc import imsave

from .local_utils import imshow, writeImg, normalize_img
_port = 8899

def display_loss(steps, values, plot=None, name='default', legend= None, port=_port):
    if plot is None:
        plot = Visdom(port=port)
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps  = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    win = name + '_loss'
    res = plot.line(
            X = steps,
            Y=  values,
            win= win,
            update='append',
            opts=dict(title = win, legend=legend),
            env = name
        )
    if res != win:
        plot.line(
            X=steps,
            Y=values,
            win=win,
            opts=dict(title=win, legend=legend),
            env=name
        )

class plot_scalar(object):
    def __init__(self, name='default', env='main', rate= 1, handler=None, port = _port):

        self.__dict__.update(locals())
        self.values = []
        self.steps = []
        if self.handler is None:
            self.handler = Visdom(port=port)
        self.count = 0
        
    def plot(self,values, step = None):
        org_type_chk = type(values) is  list
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
        steps  = np.array(repeat_steps).transpose()
        values = np.array(self.values).transpose()
        
        assert not np.isnan(values).any(), 'nan error in loss!!!'
        res = self.handler.line(
                X = steps,
                Y=  values,
                win= self.name,
                update='append',
                opts=dict(title = self.name, legend=None),
                env = self.env
            )

        if res != self.name:
            self.handler.line(
                X=steps,
                Y=values,
                win=self.name,
                env=self.env,
                opts=dict(title=self.name, legend=None)
            )

        self.reset()


def plot_img(X=None, win= None, env=None, plot=None,port=_port):
    if plot is None:
        plot = Visdom(port = port)
    if X.ndim == 2:
        plot.heatmap(X=np.flipud(X), win=win, #
                 opts=dict(title=win), env=env)
    elif X.ndim == 3:
        # X is BWC
        norm_img = normalize_img(X)
        plot.image(norm_img.transpose(2,0,1), win=win,
                   opts=dict(title=win), env=env)

def display_timeseries(strumodel, BatchData, BatchLabel, plot=None, name='default', port=_port):
    if plot is None:
        plot = Visdom(port=port)
    B, T, C, W, H = BatchData.shape
    pred_T = BatchLabel.shape[1]

    batch_id =  random.randint(0, B-1)
    intv = 9

    data_len = min(10, T)
    pred_len = min(10, pred_T)
    #sel_t = random.randint(0, T-len)

    inputdata  = BatchData[batch_id:batch_id+1,...]
    prediction = strumodel.predict(inputdata)
    labeldata  = BatchLabel[batch_id:batch_id+1,...]

    images_shape = (W, int( data_len*(H+intv) - intv  ))
    label_shape   = (W, int( pred_len*(H+intv) - intv  ))
    batch_content = np.zeros(images_shape)
    predict_content = np.zeros(label_shape)
    label_content = np.zeros(label_shape)
    #fill the image
    for idx in range(data_len):
        rs, cs = 0, idx*(H+intv)
        batch_content[rs:rs+W,  cs:cs+H]   = inputdata[0,idx,0]
       
    for idx in range(pred_len):
        rs, cs = 0, idx*(H+intv)
        predict_content[rs:rs+W,  cs:cs+H] = prediction[0,idx,0]
        label_content[rs:rs+W,  cs:cs+H]   = labeldata[0,idx,0]

    diff_abs = np.abs(predict_content - label_content)  # we do this intentionally to do sanity check
    #diff_ratio = diff_abs/(np.abs(label_content) + 1)

    #imshow(predict_content)
    #imshow(label_content)
    plot.heatmap(X = np.flipud(batch_content), win = name + '_OriginalImage',
           opts=dict(title = name + '_OriginalImage'), env = name)
    plot.heatmap(X = np.flipud(label_content), win = name + '_GroundTruth',
           opts=dict(title = name + '_GroundTruth'), env = name)
    plot.heatmap(X = np.flipud(predict_content), win = name + '_Prediction',
           opts=dict(title = name + '_Prediction'), env = name)
    #plot.heatmap(X = diff_abs, win = name + '_diff_abs',
    #       opts=dict(title = name + '_diff_abs'), env = name)
    #plot.heatmap(X = diff_ratio, win = name + '_diff_ratio',
    #       opts=dict(title = name + '_diff_ratio'), env = name)

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
                X = X.transpose(0,2,3,1)
            h, w, c = X[0].shape[:3]
            hgap, wgap = int(0.1*h), int(0.1*w)
            img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap,c))
        elif X.ndim == 3:
            h, w = X[0].shape[:2]
            hgap, wgap = int(0.1*h), int(0.1*w)
            img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw - wgap))
        else:
            assert 0, 'you have wrong number of dimension input {}'.format(X.ndim) 
        for n, x in enumerate(X):
            i = n%nw
            j = n // nw
            rs, cs = j*(h+hgap), i*(w+wgap)
            img[rs:rs+h, cs:cs+w] = x
        if c == 1:
            img = img[:,:,0]
        if save:
            writeImg(normalize_img(img), save_path)
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
            for idx, X in enumerate(X_list) :       
                X_list[idx] = X.transpose(0,2,3,1)
        
        X = X_list[0]
        h, w, c = X[0].shape[:3]
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap,c))

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
        img = img[:,:,0]
    if save:
        save_image = (img.copy() + 1) /2 * 255 
        writeImg(save_image.astype(np.uint8), save_path)
    return img