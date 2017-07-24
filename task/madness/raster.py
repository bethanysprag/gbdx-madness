"""Raster class. Contains methods for manipulating raster data.

Everything could be better tested and more robust to handle a wider
assortment of edge cases. But it's a start. The purpose is to abstract
a few things so that one can easily clip, reproject, resample and
have the results available as numpy arrays with a minimal
amount of typing.

.. testsetup:: *

    from pcross.raster import Raster
    r1 = Raster.fromFile('./testdata/30m_20150825_164853_0b09.tif')

.. testcleanup:: *

    import os
    os.remove('test_file.tif')

"""

import sys
from itertools import chain, islice

import numpy as np

try:
    from osgeo import gdal, gdalnumeric, ogr, osr

except ImportError, e:
    import gdal
    import ogr
    import gdalnumeric
    import osr


gdal.UseExceptions()
ogr.UseExceptions()


# Helper function to chunk up an iterable for our parmap
# http://stackoverflow.com/a/24527424/1868286
def chunks(iterable, size=100):
    """Create iterable chunks of a iterable.

    :param iterable: The iterable to chunk.
    :type iterable: iterable
    :param size: Size of each chunk.
    :type size: int
    :returns: An iterable of length size.

    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


class Raster(object):
    """Simple raster base class to make dealing with raster data
    less painful.

    :param ds: A GDALDataset.
    :type ds: GDALDataset

    """

    _GDAL2NP_CONVERSION = {
        1: np.uint8,
        2: np.uint16,
        3: np.int16,
        4: np.uint32,
        5: np.int32,
        6: np.float32,
        7: np.float64,
        10: np.complex64,
        11: np.complex128,
    }

    _NP2GDAL_CONVERSION = {
        'uint8': 1,
        'uint16': 2,
        'int16': 3,
        'uint32': 4,
        'int32': 5,
        'float32': 6,
        'float64': 7,
        'complex64': 10,
        'complex128': 11,
    }

    def __init__(self, ds):
        self._ds = ds
        # Get the geotransform
        self._gt = self._ds.GetGeoTransform()

        # Get the wkt
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self._ds.GetProjectionRef())
        self._geom = srs.ExportToWkt()

        # get epsg code
        self._epsg = int(srs.GetAttrValue("AUTHORITY", 1))

        # Set the extent
        self._extent = self._getExtent()

        # set the data type
        self._dtype = self._GDAL2NP_CONVERSION[
            self._ds.GetRasterBand(1).DataType]

        self._array = None

    @property
    def ds(self):
        """GDAL dataset

        :returns: GDALDataset -- a instance of a gdal object.

        .. doctest::

            >>> from pcross.raster import Raster
            >>> r1 = Raster.fromFile('./testdata/30m_20150825_164853_0b09.tif')
            >>> type(r1.ds)
            <class 'osgeo.gdal.Dataset'>

        """
        return self._ds

    @property
    def dtype(self):
        """Data type of raster

        :returns: string

        .. doctest::

            >>> print r1.dtype
            <type 'numpy.uint16'>

        """
        return self._dtype

    @property
    def array(self):
        """Dataset as numpy array


        :returns: np.array -- n x m x num bands numpy array.

        .. doctest::

            >>> type(r1.array)
            <type 'numpy.ndarray'>

            # The array element is y, x, num bands
            >>> r1.array.shape
            (434, 439, 4)

            # rather than gdal. z, y, x
            >>> import gdal
            >>> r1.ds.ReadAsArray().shape
            (4, 434, 439)

        """
        # return the array
        if self._array is None:
            self._array = self._toArray(self._dtype)
        return self._array

    @property
    def epsg(self):
        """Get the epsg code of a raster

        :returns: int -- the epsg

        .. doctest::

            >>> r1.epsg
            32616

        """

        return self._epsg

    @property
    def geom(self):
        """Well known text representation of the
        geometry of the raster.

        :returns: string -- well known text.

        .. doctest::

            >>> print r1.geom  # doctest: +ELLIPSIS
            PROJCS["WGS 84 / UTM zone 16N",GEOGCS["WGS...

        """
        return self._geom

    @property
    def extent(self):
        """Tuple of coordinates for the corners of the extent.

        :returns: tuple --  (minx, miny, maxx, maxy)


        .. doctest::

            >>> type(r1.extent)
            <type 'tuple'>

            >>> r1.extent
            (366963.0, 4368678.0, 380133.0, 4381698.0)

        """
        return self._extent

    @property
    def gt(self):
        """Geotransform of the raster

        :returns: tuple -- the geotransform.

        .. doctest::

            >>> r1.gt
            (366963.0, 30.0, 0.0, 4381698.0, 0.0, -30.0)

        """
        return self._gt

    def __eq__(self, raster):
        """Check to see if two rasters are 'equal'.

        Compares two raster objects to ensure that they
        are the same in every way. Checks data type,
        extent, geometry and geotransform.

        :param raster: Raster instance to compare to.
        :type raster: Raster
        :returns: bool -- True if successful, False otherwise.

        .. doctest::

            >>> r2 = Raster.fromFile('./testdata/30m_20150825_164853_0b09.tif')
            >>> r1 == r2
            True

            >>> np.array_equal(r1.array, r2.array)
            True

            >>> r3=Raster.fromFile('./testdata/SUB_LC80230332015237LGN00.tif')
            >>> r1 == r3
            False

        """
        if not np.array_equal(self.array, raster.array):
            return False

        for prop in ['_dtype', '_extent', '_geom', '_gt']:
            if not (vars(self)[prop] == vars(raster)[prop]):
                return False

        return True

    def _toArray(self, dtype):
        """Turns a gdal dataset into a numpy array.

        :param dtype: The numpy data type of the raster.
        :returns: np.array -- array ordered for easy viewing.
        """
        numBands = self._ds.RasterCount
        x = self._ds.RasterXSize
        y = self._ds.RasterYSize

        # pre allocate
        out = np.empty((y, x, numBands)).astype(dtype)

        for i in range(numBands):
                out[:, :, i] = self._ds.GetRasterBand(i+1).ReadAsArray()
        return out

    def _getExtent(self):
        """Get the extent of the raster

        :returns: tuple -- (minx, miny, maxx, maxy)
        """
        gt = self.gt
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1]*self.ds.RasterXSize
        miny = maxy + gt[5]*self.ds.RasterYSize
        return (minx, miny, maxx, maxy)

    # The following are helper methods, mostly
    # required for clipping and subsetting
    # geotiffs.

    def _worldToPixel(self, gt, x, y):
        """Get the (pixel,line) from a long, lat.
        modified from:
        http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

        :param geoMatrix: Gdal Geotransform
        :type name: tuple
        :param x: longitude.
        :type x: float
        :param y:  lattitude.
        :type y: float
        :returns:  Tuple -- the (pixel, line).

        """
        ulX = gt[0]
        ulY = gt[3]
        xDist = gt[1]
        # pixel = int((x - ulX) / xDist)
        # line = int((ulY - y) / xDist)
        pixel = int(round((x - ulX) / xDist))
        line = int(round((ulY - y) / xDist))
        return (pixel, line)

    def _openArray(self, array, xoff=0, yoff=0):
        """
        modified from:
        http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

        :param array: Numpy array of raster data.
        :type array: np.array
        :param xoff: Offset in the X direction.
        :type xoff: int
        :param yoff: Offset in the Y direction.
        :type yoff: int

        """
        ds = gdal.Open(gdalnumeric.GetArrayFilename(array))
        gdalnumeric.CopyDatasetInfo(self.ds,
                                    ds,
                                    xoff=xoff,
                                    yoff=yoff)
        return ds

    def clip(self, extent):
        """Clip raster to some extent.

        :param extent: The extent to clip. (minx, miny, maxx, maxy)
        :type extent: tuple
        :returns: A clipped Raster instance.

        .. doctest::

            >>> r2 = Raster.fromFile(
            ... './testdata/SUB_LC80230332015237LGN00.tif')
            >>> r1.array.shape == r2.array.shape
            False

            >>> r2.extent
            (363545.11328763305, 4363252.075809084, 386881.3006813806,...

            >>> r2_clip = r2.clip(r1.extent)
            >>> r2_clip.extent
            (366964.5546281051, 4368680.93748927, 380132.4032988701,...

            >>> r2_clip.array.shape == r1.array.shape
            False

            >>> r2_clip.array.shape
            (434, 439, 3)

            >>> r1.array.shape
            (434, 439, 4)

            >>> r3 = Raster.fromFile(
            ... './testdata/CLIP_SUB_LC80230332015237LGN00.tif')
            >>> r3 == r2_clip
            True

        """
        return self.round(multiple=None, extent=extent)

    def round(self, multiple=10, extent=None):
        """Round the size of raster to multiple of ten pixels"""

        src_array = self.ds.ReadAsArray()

        src_gt = self.gt

        if extent is None:
            minx, miny, maxx, maxy = self.extent
        else:
            minx, miny, maxx, maxy = extent

        ulx, uly = self._worldToPixel(src_gt, minx, maxy)
        lrx, lry = self._worldToPixel(src_gt, maxx, miny)

        # round to multiple
        if multiple is not None:
            offsetx = int(lrx / multiple) * multiple
            offsety = int(lry / multiple) * multiple
            ulx = lrx - offsetx
            uly = lry - offsety

        try:
            clip = src_array[:, uly:lry, ulx:lrx]
        except IndexError:
            clip = src_array[uly:lry, ulx:lrx]

        driver = gdal.GetDriverByName('MEM')
        dst_ds = driver.CreateCopy(
            '',
            self._openArray(
                clip,
                xoff=ulx,
                yoff=uly)
        )

        return Raster(dst_ds)

    def getIntersection(self, im2):
        """
        Gets the spatial intersection between self and another
        Raster instance. Should be in the same projections.

        :param im2: Image to intersect with.
        :returns: The intersection -- (ulx, uly, llx, lly) # wrong

        """
        if self.epsg != im2.epsg:
            return None

        ds = [self.ds, im2.ds]
        gts = [x.GetGeoTransform() for x in ds]

        rs = [[gt[0],  # Top left X
               gt[3],  # Top left Y
               gt[0] + (gt[1] * d.RasterXSize),  # Top left X + width
               gt[3] + (gt[5] * d.RasterYSize)]  # Top left Y - height
              for gt, d in zip(gts, ds)]

        # intersection coordinates for the top left,
        # bottom right of each dataset.
        intersection = [max(rs[0][0], rs[1][0]),  # 0, top left x, minx
                        min(rs[0][1], rs[1][1]),  # 1, top left y, maxy
                        min(rs[0][2], rs[1][2]),  # 2, bottom right x, maxx
                        max(rs[0][3], rs[1][3])]  # 3, bottom right y, miny

        minx, miny = intersection[0], intersection[3]
        maxx, maxy = intersection[2], intersection[1]
        return (minx, miny, maxx, maxy)

    def fishnet(self, height, width):
        """Create a fishnet for the image"""
        minx, miny, maxx, maxy = self.extent
        rows = np.ceil((maxy-miny)/height)
        cols = np.ceil((maxx-minx)/width)

        ringXleftOrigin = minx
        ringXrightOrigin = minx + width
        ringYtopOrigin = maxy
        ringYbottomOrigin = maxy - height

        col_idx = 0
        while col_idx < cols:
            col_idx += 1
            ringYtop = ringYtopOrigin
            ringYbottom = ringYbottomOrigin
            row_idx = 0

            while row_idx < rows:
                row_idx += 1
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(ringXleftOrigin, ringYtop)
                ring.AddPoint(ringXrightOrigin, ringYbottom)
                ring.AddPoint(ringXleftOrigin, ringYbottom)
                ring.AddPoint(ringXleftOrigin, ringYtop)
                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                ringYtop = ringYtop - height
                ringYbottom = ringYbottom - height
                # We want (minx, miny, maxx, maxy)
                minX, maxX, minY, maxY = poly.GetEnvelope()
                yield minX, minY, maxX, maxY

            ringXleftOrigin = ringXleftOrigin + width
            ringXrightOrigin = ringXrightOrigin + width

    def multiClip(self, features):
        """take a list of extents, features, polygons, etc.
           yield a raveled arrays. """

        src_array = self.ds.ReadAsArray()

        src_gt = self.gt

        for feat in features:
            minx, miny, maxx, maxy = feat

            ulx, uly = self._worldToPixel(src_gt, minx, maxy)
            lrx, lry = self._worldToPixel(src_gt, maxx, miny)

            try:
                clip = src_array[:, uly:lry, ulx:lrx]
            except IndexError:
                clip = src_array[uly:lry, ulx:lrx]

            yield clip.reshape(-1, 1)

    def _warpCallback(self, pct, message, user_data):
        """callback for reproject progress"""

        sys.stdout.write('\r')
        sys.stdout.write("[%-100s] %d%%" % ('='*int(pct*100), int(pct*100)))
        sys.stdout.flush()
        return 1

    def warp(self, epsg, pixel_size, resampling='average', progress=False):
        """Reproject a raster and resample it.

        inspired by: https://jgomezdans.github.io/gdal_notes/reprojection.html
        and: warp_27 @ https://svn.osgeo.org/gdal/trunk/autotest/alg/warp.py

        Resmpling methods correspond to intrinsic gdal
        resampling methods. Available options are:
        'average': gdal.GRA_Average,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline,
        'lanczos': gdal.GRA_Lanczos,
        'mode': gdal.GRA_Mode,
        'nearest': gdal.GRA_NearestNeighbour

        :param epsg: The epsg code to reproject to
        :type epsg: Integer
        :param pixel_size: The desired pixel size of the new raster
        :type pixel_size: Float
        :param resampling: The resampling method
        :type resampling: String
        :param progress: Show progress bar
        :type progress: Bool
        :returns: a New reprojected/resampled Raster instance

        .. doctest::

            >>> r2=Raster.fromFile('./testdata/SUB_LC80230332015237LGN00.tif')

            # Resample the raster without changing it's projection
            >>> pixel_size = 50
            >>> resampled_r2 = r2.warp(r2.epsg, pixel_size, 'nearest')
            >>> resampled_r2.epsg == r2.epsg
            True

            # The pixel size is element 1 of the geotransform
            >>> r2.gt[1] == pixel_size
            False

            >>> resampled_r2.gt[1] == pixel_size
            True

        """
        resamp = {'average': gdal.GRA_Average,
                  'bilinear': gdal.GRA_Bilinear,
                  'cubic': gdal.GRA_Cubic,
                  'cubicspline': gdal.GRA_CubicSpline,
                  'lanczos': gdal.GRA_Lanczos,
                  'mode': gdal.GRA_Mode,
                  'nearest': gdal.GRA_NearestNeighbour}

        pixel_size = float(pixel_size)
        gt = self.gt
        x_size = self.ds.RasterXSize
        y_size = self.ds.RasterYSize
        num_bands = self.ds.RasterCount

        from_crs = osr.SpatialReference()
        from_crs.ImportFromWkt(self.ds.GetProjectionRef())

        to_crs = osr.SpatialReference()
        to_crs.ImportFromEPSG(epsg)

        tx = osr.CoordinateTransformation(from_crs, to_crs)

        (ulx, uly, ulz) = tx.TransformPoint(gt[0], gt[3])
        (lrx, lry, lrz) = tx.TransformPoint(gt[0] + gt[1]*x_size,
                                            gt[3] + gt[5]*y_size)

        driver = gdal.GetDriverByName('MEM')

        dest = driver.Create('',
                             int((lrx - ulx)/pixel_size),
                             int((uly - lry)/pixel_size),
                             num_bands,
                             Raster._NP2GDAL_CONVERSION[str(self.array.dtype)])
        new_gt = (ulx,
                  pixel_size,
                  gt[2],
                  uly,
                  gt[4],
                  -pixel_size)

        dest.SetGeoTransform(new_gt)
        dest.SetProjection(to_crs.ExportToWkt())

        cbk = None
        cbk_user_data = None
        if progress:
            cbk = self._warpCallback

        error_thresh = 0.125  # same value as in gdalwarp
        gdal.ReprojectImage(self.ds,
                            dest,
                            from_crs.ExportToWkt(),
                            to_crs.ExportToWkt(),
                            resamp[resampling],
                            0.0,
                            error_thresh,
                            cbk,
                            cbk_user_data)

        return Raster(dest)

    def toFile(self, file_name, format='GTiff'):
        """Write raster to file

        :param file_name: The output file name.
        :type file_name: string
        :param format: The output file format, 'GTiff' etc.
        :type format: string
        :returns: Raster instance of the newly written file.

        .. doctest::

            >>> r1.toFile('test_file.tif', 'GTiff')
            <pcross.raster.Raster object at 0x...>

            >>> import filecmp
            >>> filecmp.cmp('test_file.tif',
            ...             './testdata/30m_20150825_164853_0b09.tif')
            True

        """

        driver = gdal.GetDriverByName(format)
        dst_ds = driver.CreateCopy(file_name, self.ds, 0)
        dst_ds = None
        del dst_ds
        return Raster.fromFile(file_name)

    @classmethod
    def stackFromRasters(cls, rasters, tol='exact'):
        """Stack a list of Raster objects into a single Raster object

        This is for landsat like situations where the bands
        are often packaged seperate. But will also work for the
        stacking of arbitrary Rasters that share the same location
        and extent. See tol.

        :param rasters: List of Raster objects to stack
        :type file_names: list of Rasters
        :param tol: the tolerance with respect to the geotransform and boundry
            coordinates. Clipping often results in slightly different precision
            with respect to coordinates. The images are effectively in the same
            place, but have slightly different boundaries. If tol='exact' this
            won't be tolerated. Any otther value will create a stack as long as
            the projections are equivalent and the array shapes are equal.
        :type tol: string
        :returns: A Raster instance n_bands = n_files

        .. doctest::

            >>> to_stack = ['./testdata/30m_20150825_164853_0b09.tif',
            ...             './testdata/CLIP_SUB_LC80230332015237LGN00.tif']
            >>> rasters = [Raster.fromFile(x) for x in to_stack]
            >>> stack = Raster.stackFromRasters(rasters)
            Traceback (most recent call last):
              ...
            ValueError: Input images have different geotransforms.

            >>> stack = Raster.stackFromRasters(rasters, tol=None)
            >>> stack.array.shape
            (434, 439, 7)

            >>> to_stack = ['./testdata/30m_20150825_164853_0b09.tif',
            ...             './testdata/30m_20150825_164853_0b09.tif']
            >>> rasters = [Raster.fromFile(x) for x in to_stack]
            >>> stack = Raster.stackFromRasters(rasters)
            >>> stack.array.shape
            (434, 439, 8)

        """
        # these things need to be identical if tol='exact'
        if tol is 'exact':
            if not len(set([x.geom for x in rasters])) <= 1:
                raise ValueError('Input images have different geometries.')

            if not len(set([x.gt for x in rasters])) <= 1:
                raise ValueError('Input images have different geotransforms.')

            if not len(set([x.extent for x in rasters])) <= 1:
                raise ValueError('Input images have different extents.')

        # These things need to match regardless
        if not len(set([x.epsg for x in rasters])) <= 1:
            raise ValueError('Input images have different epsg codes.')

        if not len(set([x.dtype for x in rasters])) <= 1:
            raise TypeError('Input images have different types.')

        # Only need to match in x,y not in z
        # rast.array.shape[0:2]
        if not len(set([x.array.shape[0:2] for x in rasters])) <= 1:
            raise ValueError('Input images have different shapes.')

        # This should all be the same
        epsg = rasters[0].epsg
        gt = rasters[0].gt
        width, height = gt[1], gt[5]
        origin = (gt[0], gt[3])
        arr = np.dstack([x.array for x in rasters])

        return cls.fromArray(epsg, origin, height, width, arr)

    @classmethod
    def stackFromFiles(cls, file_names, tol='exact'):
        """Stack a list of rasters from their files.

        Convenience function to load files into Raster objects
        before calling stackFromRasters.

        :param file_names: list of the /paths/filenames to files
        :type file_names: list of strings
        :param tol: the tolerance with respect to the geotransform and boundry
            coordinates. Clipping often results in slightly different precision
            with respect to coordinates. The images are effectively in the same
            place, but have slightly different boundaries. If tol='exact' this
            won't be tolerated. Any otther value will create a stack as long as
            the projections are equivalent and the array shapes are equal.
        :type tol: string
        :returns: A Raster instance n_bands = n_files

        .. doctest::

            >>> to_stack = ['./testdata/30m_20150825_164853_0b09.tif',
            ...             './testdata/CLIP_SUB_LC80230332015237LGN00.tif']
            >>> stack = Raster.stackFromFiles(to_stack)
            Traceback (most recent call last):
              ...
            ValueError: Input images have different geotransforms.

            >>> to_stack = ['./testdata/30m_20150825_164853_0b09.tif',
            ...             './testdata/30m_20150825_164853_0b09.tif']
            >>> stack = Raster.stackFromFiles(to_stack)
            >>> stack.array.shape
            (434, 439, 8)

        """
        rast = [Raster.fromFile(x) for x in file_names]
        return cls.stackFromRasters(rast, tol)

    @classmethod
    def fromArray(cls, epsg, origin, height, width, array):
        """Creates a raster instance from a array.

        :param epsg: The epsg code of the projection.
        :type epsg: int
        :param origin: The origin of the raster.
        :type origin: tuple of floats.
        :param height: The pixel height.
        :type height: float
        :param width: The pixel width.
        :type width: float
        :param array: The numpy array to become the raster.
        :type array: numpy array -- rows, cols, num_bands
        :returns: A Raster instance.

        .. doctest::

            >>> import numpy as np
            >>> array = np.array(
            ...     [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            ...     [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            ...     [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            ...     [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            ...     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            ...     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ...     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ...     1]]).astype(np.uint8)

            >>> origin = (-123.25745, 45.43013)
            >>> width, height = (10, 10)
            >>> epsg = 4326
            >>> fa = Raster.fromArray(epsg, origin, height, width, array)
            >>> (fa.array[:,:,0] == array).all()
            True

            # Multiband. rows, cols, num_bands
            >>> arr = r1.array
            >>> width, height = r1.gt[1], r1.gt[5]
            >>> epsg = r1.epsg
            >>> origin = (r1.gt[0], r1.gt[3])
            >>> fa_r1 = Raster.fromArray(epsg, origin, height, width, arr)

            # One would expect the following to be true, unfortunately
            # it isn't.
            >>> fa_r1 == r1
            False

            >>> fa_r1.gt == r1.gt
            True

            >>> import numpy as np
            >>> np.array_equal(r1.array, fa_r1.array)
            True

            # The reason being is that the geometries are
            # different despite all things being equal.
            >>> fa_r1.geom == r1.geom
            False

            # Despite the WKT being different, the geometries
            # appear to be equivalent. This could likely be the
            # difference between OGC WKT and ESRI WKT.
            # For a thorough discussion of this phenomenon,
            # See: http://gis.stackexchange.com/q/70281

        """
        x, y = origin

        # We have two cases, a single channel image
        # and a multi band image.
        # this obviously needs to be straightened.
        # takes arrays as rows, cols, numbands
        try:
            rows, cols = array.shape
            num_bands = 1
        except ValueError:
            rows, cols, num_bands = array.shape

        # Creating in memory data set
        driver = gdal.GetDriverByName('MEM')
        ds = driver.Create('',
                           cols,
                           rows,
                           num_bands,
                           cls._NP2GDAL_CONVERSION[str(array.dtype)])
        ds.SetGeoTransform((x, width, 0, y, 0, height))

        # The two cases again, single band
        # or multiband.

        if num_bands == 1:
            band = ds.GetRasterBand(1)
            band.WriteArray(array)
        elif num_bands > 1:
            for idx in range(num_bands):
                band = ds.GetRasterBand(idx+1)
                band.WriteArray(array[:, :, idx])

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(epsg)
        ds.SetProjection(out_srs.ExportToWkt())
        band.FlushCache()
        return Raster(ds)

    @classmethod
    def fromFile(cls, file_name):
        """Create raster instance from a raster file.

        :param file_name: The file name of the raster to load.
        :type file_name: string
        :returns: A Raster instance.

        .. doctest::

            >>> r2 = Raster.fromFile('./testdata/30m_20150825_164853_0b09.tif')
            >>> r2 == r1
            True

        """
        try:
            ds = gdal.Open(file_name)
            return Raster(ds)
        except (RuntimeError, TypeError, NameError) as e:
            raise e
