import logging
import os
import subprocess
from difflib import SequenceMatcher
from sklearn.mixture import GaussianMixture
from functools import partial
from itertools import izip

import mgrs
import geojson
from geojson import Point, Feature, FeatureCollection

import numpy as np

from madness.mad import mad
from madness.raster import Raster

try:
    from osgeo import gdal
except ImportError:
    import gdal

DEVNULL = open(os.devnull, 'wb')


def pixelcoord(col, row, gt):
    c, a, b, f, d, e = gt
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)


def geojsonify(binary_image, precision):
    ds = Raster.fromFile(binary_image)

    # make sure it's wgs84
    # and reload
    if ds.epsg != 4326:
        cmd = ("gdalwarp -t_srs EPSG:4326 %s %s" %
               (binary_image, 'WGS84_' + binary_image))
        wrap_subprocess(cmd)

        binary_image = 'WGS84_' + binary_image
        ds = Raster.fromFile(binary_image)

    # collect some useful values
    gt = ds.gt
    m = mgrs.MGRS()
    ys, xs, _ = ds.array.nonzero()

    _pc = partial(pixelcoord, gt=gt)
    pc = np.vectorize(_pc)
    new_xs, new_ys = pc(xs, ys)

    mgr = np.vectorize(partial(m.toMGRS, MGRSPrecision=precision))
    mgrs_codes = mgr(new_ys, new_xs)
    codes, counts = np.unique(mgrs_codes, return_counts=True)

    latlons = (m.toLatLon(code) for code in codes)
    out = FeatureCollection([Feature(
        id=i,
        geometry=Point((ll[1], ll[0])),
        properties={"count": c, "mgrs": mg}) for i, ll, c, mg
        in izip(xrange(len(counts)), latlons, counts, codes)])

    out_name = binary_image[:-4] + '.JSON'
    with open(out_name, 'w') as outfile:
        geojson.dump(out, outfile)

    return True


def polygonize(in_raster):
    out_polygons = in_raster[:-4] + '.GEOJSON'
    cmd = 'gdal_polygonize.py -q -8 -f GeoJSON ' + \
        in_raster + ' ' + \
        out_polygons

    wrap_subprocess(cmd)
    return out_polygons


def cluster(in_raster):
    try:
        ds = Raster.fromFile(in_raster)
        x, y, z = ds.array.shape

        X = ds.array.reshape((x*y, z))
        gmm = GaussianMixture(n_components=2,
                              covariance_type='diag').fit(X)

        out = gmm.predict(X).reshape((x, y)).astype(np.uint8)

        num_nonzero = np.count_nonzero(out)
        num_elements = np.product(out.shape)
        num_zeros = num_elements - num_nonzero

        # we assume that most things havent changed
        # so we find the change class by finding out which is dominate
        # we want zeros to be nochange
        if num_zeros < num_nonzero:
            # this means change has been assigned to zero
            out = 1 - out

        # collect data to write file
        width, height = ds.gt[1], ds.gt[5]
        epsg = ds.epsg
        origin = (ds.gt[0], ds.gt[3])

        new = Raster.fromArray(epsg,
                               origin,
                               height,
                               width,
                               out)
        # set nodata
        band = new.ds.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.FlushCache()
        band = None

        base_name = os.path.basename(in_raster)
        new_name = 'RULE_' + base_name
        r = new.toFile(new_name)
        del r
        return new_name
    except IndexError:
        return None


def madly(image_pair):
    logger = logging.getLogger("")

    temp1, temp2 = image_pair
    # load overlapping rasters
    r1 = Raster.fromFile(temp1)
    r2 = Raster.fromFile(temp2)

    # set the number of bands to look at
    num_bands = np.min([r1._ds.RasterCount, r2._ds.RasterCount])

    # TODO: something about potential alpha channels
    arr1 = r1.array[:, :, :num_bands-1]
    arr2 = r2.array[:, :, :num_bands-1]

    # check to see how "full" these tiles are
    # lets only consider 90% full tiles
    num_elements1 = np.product(arr1.shape) * 1.0
    num_elements2 = np.product(arr2.shape) * 1.0

    arr1z = np.count_nonzero(arr1) / num_elements1
    arr2z = np.count_nonzero(arr2) / num_elements2

    logger.info('Pair: %s : %.2f filled, %s : %.2f filled' %
                (temp1, (arr1z * 100), temp2, (arr2z * 100)))

    if arr1z < 0.8 or arr2z < 0.8:
        return None

    try:
        _, _, out_mads, chisq = mad(arr1, arr2)

        # get some information to write out the results
        # to a geotiff using t1 image.

        width, height = r1.gt[1], r1.gt[5]
        epsg = r1.epsg
        origin = (r1.gt[0], r1.gt[3])

        r3 = Raster.fromArray(epsg, origin, height, width, out_mads)
        # r4 = Raster.fromArray(epsg, origin, height, width, chisq)

        # saving out the difference image
        im_names = [os.path.splitext(os.path.basename(b))[0]
                    for b in image_pair]

        # we want the shared portion of the two names
        match = SequenceMatcher(
            None,
            im_names[0],
            im_names[1]).find_longest_match(0,
                                            len(im_names[0]),
                                            0,
                                            len(im_names[1]))

        out_suffix = im_names[0][match.a: match.a + match.size]
        out_name = 'MAD' + out_suffix + '.TIF'
        r3.toFile(out_name)
        # r4.toFile('CHISQ' + out_suffix + '.TIF')
        return out_name
    except (FloatingPointError, ValueError):
        return None


def tileize(in_raster, tile_size, prefix):
    dset = gdal.Open(in_raster)
    # get dimensions
    width = dset.RasterXSize
    height = dset.RasterYSize

    for i in xrange(0, width, tile_size):
        for j in xrange(0, height, tile_size):
            w = min(i + tile_size, width) - i
            h = min(j + tile_size, height) - j

            # we are going to generate tifs
            out_name = prefix + "_" + str(i) + "_" + str(j) + ".VRT"

            cmd = "gdal_translate -of VRT -srcwin " + str(i) + ", " \
                + str(j) + ", " + str(w) + ", " + str(h) + " " \
                + in_raster + " " + out_name

            yield (cmd, out_name)


def wrap_subprocess(cmd):
    return subprocess.call([cmd],
                           shell=True,
                           stdout=DEVNULL,
                           stderr=subprocess.STDOUT)


def gwarp(nps, epsg, extent, infile, outfile):
    ulx, uly, llx, lly = extent
    cmd = ("gdalwarp -multi -of VRT -tap %s -tr %f %f -te %f %f %f %f %s %s" %
           (epsg, nps, nps, ulx, uly, llx, lly, infile, outfile))
    return wrap_subprocess(cmd)
