from __future__ import annotations
import numpy as np
from PIL import Image
from astropy.io import fits as pf
import os

class TransformImage:
    """Image manipulation class.
    """
    __exts = ['png', 'jpg', 'jpeg', 'bmp']
    __transforms = ['ss', 'xy', 'cc', 'cx', 'sq']
    def __init__(self, data: np.ndarray):
        """Generate a new instance of TransformImage class from data array.

        Args:
            data (np.ndarray): Image data array. Must be 2D.

        Raises:
            TypeError: Input data must be a 2D array.
        """
        if data.ndim != 2:
            raise TypeError('Must be a 2D data array.')
        self._supersample = 1  # not supersampled
        self._transforms = []  # empty list of transforms
        self._dtype = data.dtype
        self._orig = np.copy(data)
        self._data = np.copy(self._orig)

    def reload_data(self, data: np.ndarray):
        """Load new data into this instance of TransformImage to reapply transforms.

        Args:
            data (np.ndarray): Image data array. Must be 2D.

        Raises:
            TypeError: Input data must be a 2D array.
        """
        self._dtype = data.dtype
        self._orig = np.copy(data)
        self._data = np.copy(self._orig)

    @classmethod
    def fromfile(cls, fname: str, unit: int = 0, dtype: np.dtype = float)->TransformImage:
        """Load an image from png, jpg, bmp or fits file and create a new
        instance of TransformImage class.

        Args:
            fname (str): File name to load image from.
            unit (int, optional): FITS unit number. Defaults to 0.
            dtype (np.dtype, optional): Load image as data type. Defaults to float.

        Raises:
            TypeError: Invalid file type.
            RuntimeError: Source does not exist/is a directory.

        Returns:
            TransformImage: Instance of TransformImage class.
        """
        if os.path.exists(fname) and not os.path.isdir(fname):
            ext = fname.rsplit('.', 1)[-1]
            if not (ext.lower() in TransformImage.__exts or 'fit' in ext.lower()):
                raise TypeError(
                    'Extension %s is not valid for file %s.' % (ext, fname))

            if ext.lower() in TransformImage.__exts:
                data = np.asarray(Image.open(fname), dtype=dtype)
            elif 'fit' in ext.lower():
                with pf.open(fname) as f:
                    data = np.asarray(f[unit].data, dtype=dtype)
        else:
            raise RuntimeError('%s does not exist/is a directory.' % (fname))
        return cls(data)

    def supersample(self, res: int = 16) -> None:  # supersample image
        """Supersample the image.

        Args:
            res (int, optional): Supersampling ratio [2, 32]. Defaults to 16.

        Raises:
            RuntimeError: Invalid subsampling request.
        """
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
        """Downsample the image to source sampling.
        """
        outshape = Image.fromarray(self._orig).size
        out = Image.fromarray(self._data).resize(outshape, resample=Image.BOX)
        out = np.asarray(out, dtype=self._dtype)
        self._supersample = 1
        self._data = out

    def reset(self):
        """Revert all manipulations.
        """
        self._supersample = 1
        self._transforms = []
        self._data = np.copy(self._orig)

    @property
    def data(self) -> np.ndarray:
        """Get the image data.

        Returns:
            numpy.ndarray: Image data array.
        """
        return np.asarray(self._data)

    @property
    def samplerate(self) -> int:
        """Current sample rate of the image.

        Returns:
            int: Sample rate.
        """
        return self._supersample

    def rotate(self, ang: float, center: tuple | None = None) -> None:
        """Rotate image.

        Args:
            ang (float): Rotation angle in degrees.
            center (tuple | None, optional): Rotation center, set to None to 
            rotate about the center of the image. Image origin is at top left. 
            Refer to Pillow documentation for more info.. Defaults to None.
        """
        out = Image.fromarray(self._data)
        out = out.rotate(ang, center=center)
        self._data = np.asarray(out, dtype=self._dtype)
        if center is None:
            xform = ('cc', ang, center)
        else:
            xform = ('cx', ang, center)
        self._transforms.append(xform)

    def undo(self):
        """Undo the last transformation.
        """
        # 1. supersample image again
        # this resets to original supersample
        self.supersample(self._supersample)
        xforms = self._transforms
        self._transforms = []
        if len(xforms) > 0:
            for xform in xforms[:-1]:
                self.transform(xform[0], xform[1:])

    def translate(self, tx: float = 0, ty: float = 0) -> None:
        """Translate the image.

        Args:
            tx (float, optional): Translate in the X axis. Defaults to 0.
            ty (float, optional): Translate in the Y axis. Defaults to 0.
        """
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
        """Squeeze/zoom the image.

        Args:
            sq_factor (float): Squeeze/zoom factor in fraction. < 1 to make the image smaller, > 1 to make the image larger.

        Raises:
            ValueError: Squeeze factor <= 0 not allowed.
        """
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
        """General transformation function.

        Args:
            code (str): Transformation command code (ss: Supersample, xy: Translation, cc: Rotation about center of image, cx: Rotation about given center, sq: Squeeze/zoom.)
            Additional arguments must be supplied to go with the command.

        Raises:
            TypeError: Invalid transformation command.
        """
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
        """Reapply all transforms after reset.
        """
        for xform in self.__transforms:
            self.transform(xform[0], xform[1:])

    def simplify_transforms(self)->list:
        """Simplify transformation by coalescing consecutive commands.

        Returns:
            list: List of simplified transformation commands.
        """
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
        """Save image with all transformations.

        Args:
            fname (str): Path to file.

        Raises:
            RuntimeError: Path is a directory.
        """
        if os.path.exists(fname) and os.path.isdir(fname):
            raise RuntimeError('%s is a directory'%(fname))
        outshape = Image.fromarray(self._orig).size
        img = Image.fromarray(self._data).resize(outshape, resample=Image.BOX)
        data = np.asarray(img, dtype=float)
        if len(data.shape) == 2:
            data -= data.min()
            data /= data.max()
            data *= 65535
            data = np.asarray(data, dtype=np.uint16)
            img = Image.fromarray(data)
        else:
            data = np.asarray(data, dtype=np.uint8)
            img = Image.fromarray(data)
        img.save(fname)

    def save_transforms(self, fname: str):
        """Save transformations.

        Args:
            fname (str): File name.

        Raises:
            RuntimeError: File name is a directory.
        """
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
        """Load transformations.

        Args:
            fname (str): File name.

        Raises:
            RuntimeError: File does not exist or is a directory.
        """
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
