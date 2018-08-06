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


## Running Mandala on GBDX

This version of Mandala Change Detection is designed to be run from Digital Globe's GBDX platform. It is registered as the task __exo-mandala-change-detection:0.0.2__


#### Required Inputs:

| Input_Name | Description |
|:------------|:-------------|
|image1 | Path to the first image in s3.  Either a tif, or a folder containing a single MSI tif.  Images should be orthorectified MSI < 15 degrees off-NADIR and <10% cloud cover|
 |image2 | Path to the second image.  Either a tif, or a folder containing a single MSI tif.  Images should be orthorectified MSI < 15 degrees off-NADIR and <10% cloud cover|
|data | The output data directory|

#### Optional Inputs:

| Input_Name | Description |
|:------------|:-------------|
|Polygons | Set to 'yes' to output filtered results as polygons|
|Polygon_Filter_Value | Density count to use as filter for polygon results.  Default is 20
|Polygon_Grid_Size | Grid size to use as filter for polygon results in image units.  Default is 0.0001 decimal degrees |

#### Standard Outputs:
| Output_Name | Description |
|:------------|:-------------|
|IM1_x_y.TIF | Image1 tiles, generated as part of preprocessing but useful to overlay the products on.|
|IM2_x_Y.TIF | Image2 tiles, generated as part of preprocessing but useful to overlay the products on.|
|RULE_MAD_x_y.TIF | PRODUCT.  This is a binary mask of change for the respective x/y tile at the full resolution of the image image|
|RULE_MAD_x_y.JSON | PRODUCT - This is an point vector layer for the respective x/y tile, aggregating how many pixels were flagged as containing change within a given MGRS grid cell.  This has several uses.  It can be used to generate heat maps indicating areas of dense change.  It can be used for GIS queries in order to filter out possible speckling or noise.|
|RULE_MAD_x_y_Polygons.json | PRODUCT - If using Answer Factory, this product becomes the sole standard output.  Optional if running outside of Answer Factory.  Filters the gridded points json file to identify all points containing change denser than the specified level, default value is 20.  The filtered points are converted to polygons.  This is the smallest file size of any outputs.  Areas contained within the polygon are identified as containing change.

#### Optional Outputs
| Output_Name | Description |
|:------------|:-------------|
|image1.vrt | TEMPORARY FILE -  an intermediate file generated during the preprocessing steps, output by toggling 'debug' mode|
|image2.vrt | TEMPORARY FILE -  an intermediate file generated during the preprocessing steps, output by toggling 'debug' mode|
|MAD_x_y.TIF | TEMPORARY FILE - an intermediate file generated during the processing, It does have analytic value as a final product, but requires manual interpretation so we don’t really consider it a product.  Its useful for debugging anomalous results. Output by toggling 'debug' mode|

#### Task Definition: 
```
{
    "containerDescriptors": [
        {
            "type": "DOCKER",
            "properties": {
                "image": "barnabassprague/gbdx-madness:latest"
            }
        }
    ],
    "name": "exo-mandala-change-detection",
    "inputPortDescriptors": [
        {
            "required": true,
            "type": "directory",
            "description": "Path to the first image in S3.  Either a tif, or a folder containing a single MSI tif.  Images should be orthorectified MSI < 15 degrees off-NADIR and <10% cloud cover",
            "name": "image1"
        },
        {
            "required": true,
            "type": "directory",
            "description": "Path to the second image in S3.  Either a tif, or a folder containing a single MSI tif",
            "name": "image2"
        },
        {
            "required": false,
            "type": "string",
            "description": "Set to 'yes' to output filtered results as polygons",
            "name": "Polygons"
        },
        {
            "required": false,
            "type": "string",
            "description": "Density count to use as filter for polygon results.  Default is 20",
            "name": "Polygon_Filter_Value"
        },
        {
            "required": false,
            "type": "string",
            "description": "Grid size to use as filter for polygon results.  Default is 0.0001 decimal degrees",
            "name": "Polygon_Grid_Size"
        }
    ],
    "version": "0.0.2",
    "outputPortDescriptors": [
        {
            "required": true,
            "type": "directory",
            "description": "Output data directory",
            "name": "data"
        }
    ],
    "properties": {
        "isPublic": true,
        "timeout": 17200
    },
    "description": "A task for running IR-MAD change detection on input rasters. Clips rasters to the intersection.  Identifies changes as raster masks and a point json file that aggregates changed pixels by MGRS grid.  Optional output of polygons containing features with density of detected changes greater than the filter value"
}
```


## GBDX Workflow

__Exo-mandala-change-detection__ is a discrete task within GBDX containing the minimum set of functional steps necessary for change detection as a single task for maximum utility in a variety of processing chains.  This means that in the majority of cases, it will be necessary to chain __exo-mandala-change-detection__ with other standard GBDX tasks.  One such workflow, after identifying suitable pairs and obtaining their catalog id,  is described below:

__Auto_Ordering__ (Bring the image online) -> __AOP_Strip_Processor__ (ACOMP and orthorectification) -> __exo-mandala-change-detection__ -> __StageDataToS3__ (write the results to the customers s3 bucket)


#### Sample Workflow.json file

~~~

   "name": "FromImageID2ChangeDetectionResults",
   "tasks": [
      {
         "name": "OrderStuff_1",
         "outputs": [
            {
               "name":"s3_location"
            }
            ],
         "inputs": [
            {
                "name":"cat_id",
                "value":"10100100104A0800"
            }
            ],
          "taskType": "Auto_Ordering",
          "impersonation_allowed": true
      },
      {
         "name": "OrderStuff_2",
         "outputs": [
            {
               "name":"s3_location"
            }
            ],
         "inputs": [
            {
                "name":"cat_id",
                "value":"10400100048FA500"
            }
            ],
          "taskType": "Auto_Ordering",
          "impersonation_allowed": true
      },
      {
         "name": "Preprocessing_1",
         "outputs": [
            {
               "name":"data"
            },
            {
               "name":"log"
            }
            ],
         "inputs": [
            {
                "name":"data",
                "source":"OrderStuff_1:s3_location"
            },
            {
                "name":"enable_acomp",
                "value":"true"
            },
            {
                "name":"enable_dra",
                "value":"false"
            },
            {
                "name":"enable_pansharpen",
                "value":"false"
            },
            {
                "name":"bands",
                "value":"MS"
            }
            ],
          "taskType": "AOP_Strip_Processor",
          "impersonation_allowed": true
      },
      {
         "name": "Preprocessing_2",
         "outputs": [
            {
               "name":"data"
            },
            {
               "name":"log"
            }
            ],
         "inputs": [
            {
                "name":"data",
                "source":"OrderStuff_2:s3_location"
            },
            {
                "name":"enable_acomp",
                "value":"true"
            },
            {
                "name":"enable_dra",
                "value":"false"
            },
            {
                "name":"enable_pansharpen",
                "value":"false"
            },
            {
                "name":"bands",
                "value":"MS"
            }
            ],
          "taskType": "AOP_Strip_Processor",
          "impersonation_allowed": true
      },
      {
         "name": "exo-mandala-processing",
         "outputs": [
            {
               "name":"data"
            }
            ],
         "inputs": [
            {
                "name":"image1",
                "source":"Preprocessing_1:data"
            },
            {
                "name":"image2",
                "source":"Preprocessing_2:data"
            },
            {
            	"name":"Polygons",
            	"value":"yes"
            },
            {
            	"name":"Polygon_Filter_Value",
            	"value":"20"
            }
            ],
          "taskType": "exo-mandala-change-detection",
          "impersonation_allowed": true
      },
      {
            "name": "StagetoS3_Data",
            "inputs": [
                {
                    "name": "data",
                    "source": "exo-mandala-processing:data"
                },
                {
                    "name": "destination",
                    "value": "s3://gbd-customer-data/customer_folder/Output_Directory"
                }
            ],
            "taskType": "StageDataToS3",
            "impersonation_allowed": true
        }
   ]
}

~~~