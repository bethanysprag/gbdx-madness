# Mandala Change Detection

Mandala is a fully automated change detection system with incredible speed, flexibility, and veracity. Mandala can be given two images (an image pair) from entirely different sensors with temporal deltas as great as 10 years and it will deliver accurate change detection free of false positives in a fraction of the time it would take for an analyst to do so manually.

*   Accepts image pairs with up to ten-year time deltas between observations
*   Accepts image pairs collected by different sensors and platforms
*   Accepts image pairs collected by different imaging modalities
*   Accepts image pairs collected from airborne versus satellite platforms
*   Accepts image pairs with different times of collection, look angle, solar azimuth, seasons parallax, adjacency and imaging geometry
*   Mandala change results are resistant to the effects of imaging geometry, look angle, and image artifacts
*   Mandala change results are resistant to instrument and environmental noise
*   Mandala is invariant to scaling and can run on massive datasets
*   Mandala can be integrated into many cloud or other large data platforms and service engines


## Running Mandala directly from docker container

This version of Mandala Change Detection is designed to be run from Digital Globe's GBDX platform, or in desktop mode calling the docker image directly.  The most current docker image can always be pulled from the Dockerhub repository barnabassprague/gbdx-madness:latest.  This repo is currently private, so Exogenesis partners wanting access need to submit a dockerhub username to Barnabas@exogenesis.earth to be added as a contributor:

To run the registered docker image, first login to Dockhub in the cli using credentials that have been added to the repo as a contributor.
```
docker login --username <User> --password <Password>
```
##### Syntax: 
```
$docker run -v /local_file_storage/data_location:/virtual_file_storage -it --rm barnabassprague/gbdx-madness:latest 'nongbdx' --t0 /virtual_file_storage/data_location/image1.tif --t1 /mnt/data/image2.tif --outdir /mnt/data/name_for_new_results_folder'
```
##### Example:
```
ex: $docker run -v /media/barnabas/Storage/Work/Mandala/Planet/:/mnt -it --rm barnabassprague/gbdx-madness:latest 'nongbdx' --t0 /mnt/data/image1.tif --t1 /mnt/data/image2.tif --outdir /mnt/data/results
```
### Local non-gbdx mode:
```
usage: barnabassprague/gbdx-madness:latest
                  --t0 INPUT --t1 INPUT --outdir OUTDIR
                  [--xtiles] [--ytiles] [--numcpus]


Required arguments:
  --t0			Path to the first image.  Either a tif, or a folder containing a single MSI tif.  Images should be orthorectified MSI < 15 degrees off-NADIR and <10% cloud cover
  --t1			Path to the first image.  Either a tif, or a folder containing a single MSI tif.  Images should be orthorectified MSI < 15 degrees off-NADIR and <10% cloud cover
  --outdir              The output directory

Optional arguments:
  --outdir              The output directory (REQU
  --xiles
                        Number of x tiles to subset input image into (default: 1)
  --tiles
                        Number of y tiles to subset input image into (default: 1)
  --numcpus       	Number of cpus to use. Only works if numtiles > 1 (default: 1)

```
#### Standard Outputs:
| Output_Name | Description |
|:------------|:-------------|
|IM1_x_y.TIF | Image1 tiles, generated as part of preprocessing but useful to overlay the products on.|
|IM2_x_Y.TIF | Image2 tiles, generated as part of preprocessing but useful to overlay the products on.|
|image1.vrt | TEMPORARY FILE -  an intermediate file generated during the preprocessing steps|
|image2.vrt | TEMPORARY FILE -  an intermediate file generated during the preprocessing steps|
|MAD_x_y.TIF | TEMPORARY FILE - an intermediate file generated during the processing, It does have analytic value as a final product, but requires manual interpretation so we don’t really consider it a product.  Its useful for debugging anomalous results.|
|RULE_MAD_x_y.TIF | PRODUCT.  This is a binary mask of change for the respective x/y tile at the full resolution of the image image|
|RULE_MAD_x_y.JSON | PRODUCT - This is an point vector layer for the respective x/y tile, aggregating how many pixels were flagged as containing change within a given MGRS grid cell.  This has several uses.  It can be used to generate heat maps indicating areas of dense change.  It can be used for GIS queries in order to filter out possible speckling or noise.|

