from __future__ import annotations
import numpy as np
from PIL import Image
from astropy.io import fits as pf
import os

class TransformImage:
    __exts = ['png', 'jpg', 'jpeg', 'bmp']
    __transforms = ['ss', 'xy', 'cc', 'cx', 'sq']

    def __init__(self, fname: str, unit: int = 0, dtype: np.dtype = float):
        self._supersample = 1  # not supersampled
        self._transforms = []  # empty list of transforms
        self._dtype = dtype
        if os.path.exists(fname) and not os.path.isdir(fname):
            ext = fname.rsplit('.', 1)[-1]
            if not (ext.lower() in TransformImage.__exts or 'fit' in ext.lower()):
                raise TypeError(
                    'Extension %s is not valid for file %s.' % (ext, fname))

            if ext.lower() in TransformImage.__exts:
                self._orig = np.asarray(Image.open(fname), dtype=dtype)
            elif 'fit' in ext.lower():
                with pf.open(fname) as f:
                    self._orig = np.asarray(f[unit].data, dtype=dtype)
            self._data = np.copy(self._orig)
        else:
            raise RuntimeError('%s does not exist/is a directory.' % (fname))

    def supersample(self, res: int = 16) -> None:  # supersample image
        if res < 1 or res > 32:
            raise RuntimeError('Supersample request %d invalid.' % (res))
        if res == 1:
            return
        self._supersample = res
        inval = Image.fromarray(self._orig)
        inshape = inval.size
        outshape = np.asarray(inshape, dtype=int)
        outshape[0:2] = outshape[0:2] * res
        outshape = tuple(outshape)
        out = inval.resize(outshape, resample=Image.BOX)
        out = np.asarray(out, dtype=self._dtype)
        self._data = out

    def downsample(self):  # downsample image
        outshape = Image.fromarray(self._orig).size
        out = Image.fromarray(self._data).resize(outshape, resample=Image.BOX)
        out = np.asarray(out, dtype=self._dtype)
        self._supersample = 1
        self._data = out

    def reset(self):
        self._supersample = 1
        self._transforms = []
        self._data = np.copy(self._orig)

    @property
    def data(self):
        return np.asarray(self._data)

    @property
    def samplerate(self):
        return self._supersample

    def rotate(self, ang: float, center: tuple | None = None) -> None:
        out = Image.fromarray(self._data)
        out = out.rotate(ang, center=center)
        self._data = np.asarray(out, dtype=self._dtype)
        if center is None:
            xform = ('cc', ang, center)
        else:
            xform = ('cx', ang, center)
        self._transforms.append(xform)

    def undo(self):
        # 1. supersample image again
        # this resets to original supersample
        self.supersample(self._supersample)
        if len(self.__transforms) > 0:
            for xform in self._transforms[:-1]:
                self.transform(xform[0], xform[1:])

    def translate(self, tx: float = 0, ty: float = 0) -> None:
        _tx = tx
        _ty = ty
        _tx *= self._supersample
        _ty *= self._supersample
        _tx = round(_tx)
        _ty = round(_ty)
        out = Image.fromarray(self._data)
        out = out.transform(out.size, Image.AFFINE, (1, 0, _tx, 0, 1, _ty))
        self._data = np.asarray(out, dtype=self._dtype)
        xform = ('xy', tx, ty)
        self._transforms.append(xform)

    def squeeze(self, sq_factor: float):
        if sq_factor <= 0:
            raise ValueError('Squeeze factor can not be 0 or negative')
        if sq_factor == 1:
            return
        out = Image.fromarray(self._data)
        oldsize = out.size
        newsize = tuple(np.asarray(
            np.round(np.asarray(oldsize, dtype=float)*sq_factor), dtype=int))
        out = out.resize(newsize)
        oval = np.asarray(out, dtype=float)
        # 1. balance pixel values after shrinking
        oval /= sq_factor * sq_factor
        out = np.zeros(self._data.shape, dtype=self._dtype)
        # 2. center/crop
        if sq_factor > 1:  # crop
            oldshape = out.shape
            newshape = oval.shape
            rl = (newshape[0] - oldshape[0]) // 2
            rr = rl + oldshape[0]
            cl = (newshape[1] - oldshape[1]) // 2
            cr = cl + oldshape[1]
            out = oval[rl:rr, cl:cr]
        elif sq_factor < 1:  # center
            oldshape = out.shape
            newshape = oval.shape
            rl = (oldshape[0] - newshape[0]) // 2
            rr = rl + newshape[0]
            cl = (oldshape[1] - newshape[1]) // 2
            cr = cl + newshape[1]
            out[rl:rr, cl:cr] = oval
        self._data = out
        xform = ('sq', sq_factor)
        self._transforms.append(xform)

    def transform(self, code: str, *kwargs):
        if code not in TransformImage.__transforms:
            raise TypeError('%s is not a valid transformation.' % (code))
        if code == 'xy':
            self.translate(*kwargs)
        elif code == 'cc' or code == 'cx':
            self.rotate(*kwargs)
        elif code == 'ss':
            self.supersample(*kwargs)
        elif code == 'sq':
            self.squeeze(*kwargs)

    def reapply_transforms(self):
        for xform in self.__transforms:
            self.transform(xform[0], xform[1:])

    def simplify_transforms(self):
        if len(self._transforms) <= 1:
            return self._transforms.copy()
        cmd = ''
        args = []
        out = []
        for idx, xform in enumerate(self._transforms):
            if cmd != xform[0] and cmd == '':
                cmd = xform[0]
                args = list(xform[1:])
            elif cmd == xform[0]: # same command, again -> could aggregate
                if cmd == 'ss':
                    args[0] = xform[1] # just update args if multiple supersample commands one after another, only the last one stays
                elif cmd == 'xy':
                    args[0] += xform[1] # add X-Y translations together
                    args[1] += xform[2]
                elif cmd == 'cc': # about 0
                    args[0] += xform[1]
                elif cmd == 'cx': # about different centers
                    out.append([cmd] + args)
                elif cmd == 'sq':
                    args[0] *= xform[1] # percentage product
            else: # unequal, push to out, update new
                out.append([cmd] + args)
                cmd = xform[0]
                args = list(xform[1:])
        if cmd != '':
            out.append([cmd] + args)
        return out
                
    def save_image(self, fname: str):
        if os.path.exists(fname) and os.path.isdir(fname):
            raise RuntimeError('%s is a directory'%(fname))
        img = Image.fromarray(self._data)
        img.save(fname)

    def save_transforms(self, fname: str):
        if len(self._transforms) == 0:
            return
        if os.path.exists(fname) and os.path.isdir(fname):
            raise RuntimeError('Path exists and is a directory.')
        with open(fname + '.raw', 'w') as ofile:
            ofile.write('ss,%d\n' % (self._supersample))
            for xform in self._transforms:
                outf = ''
                for xf in xform:
                    outf += str(xf) + ','
                outf = outf.rstrip()
                outf = outf.rstrip(',')
                outf += '\n'
                ofile.write(outf)
            ofile.close()
        out = self.simplify_transforms()
        with open(fname, 'w') as ofile:
            ofile.write('ss,%d\n' % (self._supersample))
            for xform in out:
                outf = ''
                for xf in xform:
                    outf += str(xf) + ','
                outf = outf.rstrip()
                outf = outf.rstrip(',')
                outf += '\n'
                ofile.write(outf)
            ofile.close()

    def load_transforms(self, fname: str):
        if not os.path.exists(fname) or os.path.isdir(fname):
            raise RuntimeError('%s does not exits/is directory')

        with open(fname, 'r') as ifile:
            for line in ifile:
                line = line.rstrip()
                words = line.split(',')
                for idx, _ in enumerate(words):
                    words[idx] = words[idx].strip()
                cmd = words[0]
                if cmd == 'ss':
                    arg = int(words[1])
                    self.transform(cmd, arg)
                elif cmd == 'cc':
                    ang = float(words[1])
                    self.transform(cmd, ang)
                elif cmd == 'cx':
                    c0 = int(words[2][1:]) # (...
                    c1 = int(words[3][:1]) # ...)
                    self.transform(cmd, ang, (c0, c1))
                elif cmd == 'xy':
                    x = float(words[1])
                    y = float(words[2])
                    self.transform(cmd, x, y)
                elif cmd == 'sq':
                    zoom = float(words[1])
                    self.transform(cmd, zoom)
