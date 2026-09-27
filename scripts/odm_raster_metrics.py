"""GDAL-side evaluation helper, executed in the pinned ODM container."""
import argparse
import json
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tif',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    dataset=gdal.Open(str(args.tif))
    if dataset is None: raise RuntimeError('Cannot open ODM orthophoto')
    if args.plan:
        plan=json.loads(args.plan.read_text())
        warped=gdal.Warp('',dataset,format='MEM',dstSRS='EPSG:4326',
                         outputBounds=plan['bounds_lonlat'],width=plan['width'],height=plan['height'],
                         resampleAlg='bilinear',dstAlpha=True,multithread=False)
        gdal.Translate(str(args.output/'common_odm.png'),warped,format='PNG')
        return
    info=gdal.Info(dataset,format='json',stats=True)
    gt=dataset.GetGeoTransform(); projection=osr.SpatialReference(wkt=dataset.GetProjection())
    geographic=osr.SpatialReference(); geographic.ImportFromEPSG(4326)
    projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr=osr.CoordinateTransformation(projection,geographic)
    corners=[]
    for x,y in ((0,0),(dataset.RasterXSize,0),(dataset.RasterXSize,dataset.RasterYSize),(0,dataset.RasterYSize)):
        px=gt[0]+x*gt[1]+y*gt[2];py=gt[3]+x*gt[4]+y*gt[5]
        corners.append(tr.TransformPoint(px,py)[:2])
    alpha=dataset.GetRasterBand(dataset.RasterCount) if dataset.RasterCount==4 else None
    covered=0
    for y in range(0,dataset.RasterYSize,512):
        a=alpha.ReadAsArray(0,y,dataset.RasterXSize,min(512,dataset.RasterYSize-y)) if alpha else np.ones((min(512,dataset.RasterYSize-y),dataset.RasterXSize))
        covered+=int(np.count_nonzero(a))
    summary=dict(width=dataset.RasterXSize,height=dataset.RasterYSize,bands=dataset.RasterCount,
                 geotransform=list(gt),projection_wkt=dataset.GetProjection(),corners_lonlat=corners,
                 covered_pixels=covered,coverage_fraction=covered/(dataset.RasterXSize*dataset.RasterYSize),
                 pixel_area_crs_units_squared=abs(gt[1]*gt[5]-gt[2]*gt[4]),
                 covered_area_crs_units_squared=covered*abs(gt[1]*gt[5]-gt[2]*gt[4]),
                 projected=bool(projection.IsProjected()),linear_units_m=projection.GetLinearUnits() if projection.IsProjected() else None)
    (args.output/'raster.json').write_text(json.dumps(summary,indent=2))
    (args.output/'gdal_info.json').write_text(json.dumps(info,indent=2))
    gdal.Translate(str(args.output/'odm.png'),dataset,format='PNG')


if __name__=='__main__':main()
