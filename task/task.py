import sys
import json
import os
import logging
from logging.handlers import RotatingFileHandler

import multiprocessing

import click
import numpy as np
try:
    import gdal
    import ogr

except:
    from osgeo import gdal, ogr

from madness.split import splits
from madness.raster import Raster

from madness.utils import (gwarp,
                           wrap_subprocess,
                           madly,
                           cluster)

# we want floating point errors
np.seterr(all='raise')
np.seterr(under='ignore')


def get_inputs(imgpath):
    """
    Get first tif in path.
    """
    for dirpath, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            # just get the first tif
            if filename.endswith('.TIF') or filename.endswith('.tif'):
                return os.path.join(dirpath, filename)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    There are two cases:
    gbdx
    Where the files will be located in places defined by the
    task specification file.
    nongbdx
    Where you pass command line options to define the inputs and
    output directory.
    """
    if ctx.invoked_subcommand is None:
        click.echo('Running in GBDX Mode')

        # load ports.json
        # currently not using
        ports_json = '/mnt/work/input/ports.json'
        default_filter = 20
	default_grid = 0.0001
        filter_value = default_filter
        grid_size = default_grid
        numcpus = multiprocessing.cpu_count()
        input_data = json.load(open(ports_json))
        try:
            debug = str(input_data['debug'])
            if debug == '':
                debug = None
        except:
            debug = None
        polygons = None
        try:
            polygons = str(input_data['Polygons'])
            if polygons != 'yes':
                polygons = None
        except:
            polygons = None
        if polygons is not None:
            try:
                filter_value = str(input_data['Polygon_Filter_Value'])
                filter_value = int(filter_value)
            except:
                filter_value = default_filter
	if polygons is not None:
            try:
                grid_size = str(input_data['Polygon_Grid_Size'])
                grid_size = float(grid_size)
            except:
                grid_size = default_grid
        main(get_inputs('/mnt/work/input/image1'),
             get_inputs('/mnt/work/input/image2'),
             '/mnt/work/output/data',
             json.load(open(ports_json)),
             xtiles=4,
             ytiles=4,
             numcpus=numcpus,
             debug=debug,
             polygons=polygons,
             filter_value=filter_value,
	     grid_size=grid_size)

    else:
        ctx.invoked_subcommand


@cli.command()
@click.option('--t0',
              default=lambda: os.environ.get('T0', ''),
              type=click.Path(exists=True),
              help='The t0 image')
@click.option('--t1',
              default=lambda: os.environ.get('T1', ''),
              type=click.Path(exists=True),
              help='The t1 image')
@click.option('--outdir',
              default=lambda: os.environ.get('OUTDIR', ''),
              help='The output directory')
@click.option('--xtiles',
              default=1,
              help='Number of x tiles to subset input image into.')
@click.option('--ytiles',
              default=1,
              help='Number of y tiles to subset input image into')
@click.option('--numcpus',
              default=1,
              help='Number of cpus to use. Only works if numtiles > 1')
@click.option('--debug',
              default=None,
              help='Leaves temp files in place for debugging')
def nongbdx(t0, t1, outdir, xtiles, ytiles, numcpus, debug='yes'):
    """
    To aid in running this as a command line docker application,
    we look for the inputs as enviromental vars.
    """
    if all([t0, t1, outdir]):
        main(os.path.abspath(t0),
             os.path.abspath(t1),
             outdir,
             None,
             xtiles,
             ytiles,
             numcpus,
	     debug='yes')
    else:
        raise ValueError("see: task.py nongbdx --help")


def JSON2Polygons(JSON_File, Polygon_file=None, threshold=30, grid_size=0.0001):
    name = names(JSON_File)
    # This is a bad code practice, but I'm removing the directory from these outputs since we've already
    # changed working directories to the output directory
    #tempRaster = '%s/TempRaster.tif' % name['directory']
    tempRaster = 'TempRaster.tif'
    if os.path.exists(tempRaster):
        os.remove(tempRaster)
    exeString = 'gdal_rasterize -burn 1 -where "count>%s" -of GTiff -tr %s %s -a_nodata 0 %s %s' % (threshold, 
                                                                                                    grid_size, 
                                                                                                    grid_size, 
                                                                                                    JSON_File, 
                                                                                                    tempRaster)
    a = os.system(exeString)
    if Polygon_file is None:
        name = names(JSON_File)
        #Polygon_file = '%s/%s_Polygons.json' % (name['directory'], name['basename'])
        Polygon_file = '%s_Polygons.json' % (name['basename'])
    exeString = 'gdal_polygonize.py -q -8 -f GeoJSON %s %s Layer Changed' % (tempRaster, 
                                                                             Polygon_file)
    b = os.system(exeString)
    if a+b != 0:
        return 'error'
    os.remove(tempRaster)
    return 0


def names(path):
    baseWExt = os.path.basename(path)
    temp = baseWExt.split('.')
    basename = temp[0]
    if len(temp) == 2:
        extension = temp[1]
    else:
        extension = ''
    folder = os.path.dirname(path)
    outName = {'basename': basename, 'extension': extension,
               'directory': folder}
    return outName


def main(img1_path, img2_path, out_dir,
         input_data, xtiles, ytiles, numcpus, debug=None, filter_value=20, polygons=None, grid_size=0.0001):

    # make and change to output directory
    try:
        os.makedirs(out_dir)
    except OSError, e:
        sys.stderr.write("Failed to create output directory, %s" % e)
        return None

    os.chdir(out_dir)

    # set up logging
    logfile = 'work.log'
    log = logging.getLogger("")
    log.setLevel(logging.DEBUG)
    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = RotatingFileHandler(
        logfile,
        maxBytes=(1048576*5),
        backupCount=7)

    fh.setFormatter(format)
    log.addHandler(fh)

    logging.info("Locating input rasters.")
    # get our tifs
    ims = []
    ims.append(img1_path)
    ims.append(img2_path)

    logging.info(
        "Finding intersection between %s and %s" % (ims[0], ims[1]))

    # load tifs to get intersection
    # we are loading the full path to the file here
    try:
        r1 = Raster.fromFile(ims[0])
        r2 = Raster.fromFile(ims[1])
    except RuntimeError, e:
        logging.info("Opening file failed, %s" % e)
        logging.info("Early exit.")
        return None

    # Find ideal pixel size (the largest between the images)
    nps = np.max([r1.gt[1], r1.gt[5], r2.gt[1], r2.gt[5]])

    # get the Intersection
    extent = r1.getIntersection(r2)
    if extent is None:
        logging.info("Images do not intersect.")
        logging.info("Early exit.")
        return None

    logging.info("Clipping rasters to same extent.")

    # we are using gdalwarp to get the intersection
    # because it's better than our codes for large files.
    # the result is a vrt.
    bases = [os.path.basename(im) for im in ims]
    vrt_names = [os.path.splitext(b)[0] + '.vrt' for b in bases]

    # should check epsg here and add it in.
    epsg = '-t_srs EPSG:%s' % str(r2.epsg)

    # from this we have new vrts
    # we are saving out into our new directory here
    check1 = gwarp(nps, epsg, extent, ims[0], vrt_names[0])
    check2 = gwarp(nps, epsg, extent, ims[1], vrt_names[1])
    if (int(check1) + int(check2) != 0):
        logging.info("Error clipping images to same extent.")
        logging.info("Early exit.")
        return None

    # clean up
    del r1, r2
    logging.info("Clipping complete.")

        # we want to load the vrt and chop out tiles from it for each raster
    # tile sizes are at most 5000 x 5000
    # 16 bit... 2 * 5000 * 5000 * 8 bands
    # seems reasonable
    # NUM_PROCESSES = multiprocessing.cpu_count()
    if numcpus == 1:
        pmap = map
    else:
        pool = multiprocessing.Pool(processes=numcpus)
        pmap = pool.map

    # Mad output is a double
    # 8 * 5000 * 5000 * 8 bands ~ 1.6GB
    # *4 = 6.4GB
    logging.info("Clipping rasters into tiles.")

    # let's you specify tile sizes in pixels rather than number
    # of tiles.
    # from madness.utils import tileize
    # im1_cmds = tileize(vrt_names[0], tile_size=5000, prefix="IM1")
    # im2_cmds = tileize(vrt_names[1], tile_size=5000, prefix="IM2")

    x_tiles = xtiles
    y_tiles = ytiles
    im1_cmds = [x for x in splits(vrt_names[0],
                                  'IM1',
                                  x_tiles,
                                  y_tiles)]

    im2_cmds = [x for x in splits(vrt_names[1],
                                  'IM2',
                                  x_tiles,
                                  y_tiles)]

    # unpcack into commands and output names
    im1_cmds, im1_tiles = zip(*im1_cmds)
    im2_cmds, im2_tiles = zip(*im2_cmds)

    logging.info("Tiling %s." % vrt_names[0])
    pmap(wrap_subprocess, im1_cmds)

    logging.info("Tiling %s." % vrt_names[1])
    pmap(wrap_subprocess, im2_cmds)

    logging.info("Tiling comlete.")

    logging.info("Finding MAD variates.")
    im_pairs = zip(im1_tiles, im2_tiles)

    mad_tiles = pmap(madly, im_pairs)
    from_mad = [x for x in mad_tiles if x is not None]

    logging.info("MAD Transformed %d tile pairs." % len(from_mad))
    logging.info("%d tile pairs omitted due to incomplete data." %
                 (len(im1_tiles) - len(from_mad)))

    # now clustering
    logging.info("Clustering MAD transformed difference images.")
    rule_ims = pmap(cluster, from_mad)
    rule_ims = [x for x in rule_ims if x is not None]

    logging.info("Clustering complete.")
    logging.info("%d MAD variates discarded due to data error" %
                 (len(from_mad) - len(rule_ims)))

    # to write polygons.
    # takes too long.
    # from madness.utils import polygonize
    # logging.info("Writing polygons from rule images.")
    # polygons = pool.map(polygonize, rule_ims)
    # logging.info("Polygons created.")

    # also takes too long.
    # write the results to a json file.
    logging.info("Aggregating change to MGRS grid.")


    # # MGRS aggregation precision
    from madness.utils import geojsonify
    from functools import partial
    # Precision 3 for RE, Pscope
    # precision 4
    gj = partial(geojsonify, precision=4)
    pmap(gj, rule_ims)

    # # close all these things
    if numcpus != 1:
        pool.close()


    #If not debug, delete  intermediate files (vrts, MAD)
    if debug is None:
        fileList = []
        deleteList = []
        for files in os.listdir(out_dir):
            fileList.append(files)
            if files.endswith('vrt'):
                deleteList.append(files)
            if files[:3] == 'MAD':
                deleteList.append(files)
        for files in deleteList:
            os.remove(files)


    if polygons is not None:
	logging.info("Writing polygons from rule images.")
        jsonList = []
        for files in os.listdir(out_dir):
            if files.endswith('JSON'):
                jsonList.append(files)
        for outJSON in jsonList:
            status = JSON2Polygons(outJSON, threshold=filter_value, grid_size=grid_size)
	    delEmpties = 1
	    if delEmpties == 1:
		logging.info("Checking polygons for null outputs.")
                polygonList = []
		removalList = []
                for files in os.listdir(out_dir):
                    if files.endswith('Polygons.json'):
                        polygonList.append(files)
                for files in polygonList:
                    #if json contains no features
		    driver = ogr.GetDriverByName('GeoJSON')
                    vs = driver.Open(files)
		    layer = vs.GetLayer()
		    count = layer.GetFeatureCount()
		    layer = None
		    vs.Destroy()
		    if count < 1:
		        removalList.append(files)
		        os.remove(files)
		        logging.info("Removing empty polygon output: %s" % files)
#		    f = open('EmptyPolygons.txt', 'w')
#                    for files in removalList:
#		        f.write('%s/n' % files)
#		    f.close()


    #DELETE FOR DEBUG ONLY
#    with open('inputData.json', 'w') as outfile:
#              json.dump(input_data, outfile)


    # write the status
    if input_data is not None:
        status = {'status': 'success', 'reason': 'task completed'}
        with open('/mnt/work/status.json', 'w') as outfile:
            json.dump(status, outfile)

    logging.info("Done.")

if __name__ == '__main__':
    cli(obj={})
