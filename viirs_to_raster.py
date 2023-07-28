import arcpy, glob, os, sys
from arcpy import env
from arcpy.sa import *
import numpy as np
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *

years = np.arange(2013, 2023, 1)

in_path = r"N:\people\spotter5\cnn_mapping\VIIRS\pts_by_year"
out_path = r"N:\people\spotter5\cnn_mapping\VIIRS\raster_by_year"
if not os.path.isdir(out_path):
    os.makedirs(out_path)

for year in years: 
    
    with arcpy.EnvManager(outputCoordinateSystem='PROJCS["NSIDC_Sea_Ice_Polar_Stereographic_North",GEOGCS["GCS_Hughes_1980",DATUM["D_Hughes_1980",SPHEROID["Hughes_1980",6378273.0,298.279411123064]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Stereographic_North_Pole"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-45.0],PARAMETER["Standard_Parallel_1",70.0],UNIT["Meter",1.0]]', cellSize=375):
        arcpy.conversion.PointToRaster(
            in_features=os.path.join(in_path, str(year) + '.shp'),
            value_field="Year",
            out_rasterdataset=os.path.join(out_path, str(year) + ".tif"),
            cell_assignment="MOST_FREQUENT",
            priority_field="NONE",
            cellsize=375,
            build_rat="BUILD"
        )
        
        print(year)


