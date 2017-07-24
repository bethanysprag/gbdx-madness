from raster import Raster


def splits(img, prefix, xtiles, ytiles):
    # (minx, miny, maxx, maxy)
    rimg = Raster.fromFile(img)
    ext = rimg.extent

    xmin = ext[0]
    xsize = ext[2] - xmin
    ysize = ext[3] - ext[1]

    xdif = xsize / xtiles

    for x in xrange(xtiles):
        xmax = xmin + xdif
        ymax = ext[3]
        ydif = ysize / ytiles

        for y in xrange(ytiles):
            ymin = ymax - ydif
            out_name = prefix + '_' + str(x) + '_' + str(y) + '.TIF'
            cmd = "gdal_translate -q \
                    -projwin %s %s %s %s \
                    -of GTiff \
                    %s \
                    %s" % (xmin,
                           ymax,
                           xmax,
                           ymin,
                           img,
                           out_name)

            yield(cmd, out_name)
            ymax = ymin
        xmin = xmax
