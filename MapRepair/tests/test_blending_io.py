import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patch_blend import blend_patch
from map_geo_checker import render_repairs,Settings
from map_io import read_map,flatten_original_extent,save_alpha_tiff,save_jpeg
from georeference import load_georeference,GeoReference


class BlendingTests(unittest.TestCase):
    def test_color_and_detail_mismatch_reduced_without_touching_outside(self):
        rng=np.random.default_rng(98)
        source=cv2.GaussianBlur(rng.uniform(35,190,(384,384,3)).astype(np.float32),(0,0),.6)
        reference=np.rint(cv2.GaussianBlur(source,(0,0),1.2)).astype(np.uint8)
        source+=np.array([14.,-9.,18.],np.float32)
        selected=np.zeros((384,384),np.uint8);selected[96:288,96:288]=1
        result,report=blend_patch(reference,source,selected,np.ones(selected.shape,np.float32),selected==0)
        d=cv2.distanceTransform(np.pad(selected,1),cv2.DIST_L2,5)[1:-1,1:-1]
        old_weight=np.clip(d/8,0,1)[...,None]
        old=np.rint(reference*(1-old_weight)+source*old_weight).clip(0,255).astype(np.uint8)
        band=(d>0)&(d<32)
        before=float(np.abs(old.astype(float)-reference)[band].mean())
        after=float(np.abs(result.astype(float)-reference)[band].mean())
        print('Appearance seam error:',dict(before=before,after=after,blur=report['blur_sigma_px']),flush=True)
        self.assertLess(after,before*.35)
        self.assertGreater(report['blur_sigma_px'],0)
        self.assertLess(report['texture_ratio_after'],report['texture_ratio_before'])
        np.testing.assert_array_equal(result[selected==0],reference[selected==0])
        identity,_=blend_patch(reference,reference,selected,np.ones(selected.shape,np.float32),selected==0)
        np.testing.assert_array_equal(identity,reference)

    def test_empty_appearance_context_is_finite_and_safe(self):
        image=np.full((80,80,3),100,np.uint8);selected=np.ones((80,80),np.uint8)
        result,report=blend_patch(image,image+5,selected,np.zeros(selected.shape,np.float32),selected==0)
        self.assertFalse(report['appearance_matched']);np.testing.assert_array_equal(result,image)

    def test_patch_order_and_original_alpha_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            rng=np.random.default_rng(89);image=rng.integers(50,170,(256,384,4),np.uint8);image[:,:,3]=255
            image[:32,:,3]=0
            owners=np.zeros((4,6),np.uint16);owners[1:3,1:3]=1;owners[1:3,3:5]=2
            patches=[]
            for i in (1,2):
                p=Path(td)/f'{i}.png';src=image.copy();src[:,:,:3]+=i*3;cv2.imwrite(str(p),src)
                patches.append(dict(file=str(p),source_index=i,tiles=4,H=np.eye(3).tolist()))
            a,_=render_repairs(image,owners,patches,(256,384),64,Settings(),lambda s:None)
            b,_=render_repairs(image,owners,patches[::-1],(256,384),64,Settings(),lambda s:None)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,:,3],image[:,:,3])


class RasterIOTests(unittest.TestCase):
    def test_tiff_band_roles_internal_mask_and_rotated_extent(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'map.tif';h,w=36,48
            samples=np.empty((4,h,w),np.uint8);samples[:]=np.array([90,70,30,255])[:,None,None]
            samples[3,:4]=0;samples[3,10:16,20:28]=128
            # Opaque black is real map content, not an inferred empty pixel.
            samples[:3,20:26,10:16]=0
            mask=np.full((h,w),255,np.uint8);mask[28:31,30:35]=0
            transform=Affine(.12,.03,500000,.02,-.12,4000000)
            with rasterio.open(p,'w',driver='GTiff',width=w,height=h,count=4,dtype='uint8',
                               photometric='RGB',alpha='YES',transform=transform,crs='EPSG:32615') as ds:
                ds.write(samples)
                ds.write_mask(mask)
            image,info=read_map(p);expected=np.minimum(samples[3],mask)
            self.assertEqual(info['alpha_band'],4);np.testing.assert_array_equal(image[:,:,3],expected)
            np.testing.assert_array_equal(image[22,12],[0,0,0,255])
            np.testing.assert_array_equal(image[12,22],[30,70,90,128])
            geo=load_georeference(p,image.shape)
            np.testing.assert_allclose(geo.matrix@np.array([0,0,1]),[*transform*(.5,.5),1])
            # Extra padded pixels must be cropped; alpha comes from the original.
            final=np.full((h*2+10,w*2+6,4),[30,70,90,255],np.uint8)
            final[40:52,20:32,:3]=0
            flattened=flatten_original_extent(final,image[:,:,3])
            self.assertEqual(flattened.shape,(h*2,w*2,3))
            np.testing.assert_array_equal(flattened[0,0],[255,255,255])
            np.testing.assert_array_equal(flattened[44,24],[0,0,0])
            np.testing.assert_allclose(flattened[24,44],[142,162,172],atol=1)
            alpha_path=Path(td)/'final_2x_alpha.tif';save_alpha_tiff(alpha_path,image[:,:,3],geo)
            with rasterio.open(alpha_path) as ds:
                np.testing.assert_array_equal(ds.read(1),np.repeat(np.repeat(expected,2,0),2,1))
                np.testing.assert_allclose(tuple(ds.transform),tuple(transform*Affine.scale(.5)),atol=1e-10)
            metadata=geo.export(td,'final_2x',2,world_suffix='.jgw')
            output_geo=GeoReference(metadata['pixel_to_world'],metadata['crs'],'test')
            np.testing.assert_allclose(output_geo.latlon([w*2-.5,h*2-.5]),geo.latlon([w-.5,h-.5]),atol=1e-10)
            save_jpeg(Path(td)/'final.jpg',flattened)
            loaded=cv2.imread(str(Path(td)/'final.jpg'));self.assertEqual(loaded.shape,flattened.shape)
            self.assertTrue((Path(td)/'final_2x.jgw').exists())

    def test_rgb_nodata_and_uint16_display_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'nodata.tif';samples=np.full((3,32,32),2048,np.uint16);samples[:,10:14,10:14]=0
            # One zero channel in a colored pixel must not invalidate that pixel.
            samples[0,18:22,18:22]=0
            with rasterio.open(p,'w',driver='GTiff',width=32,height=32,count=3,dtype='uint16',nodata=0,
                               transform=Affine(.1,0,500000,0,-.1,4000000),crs='EPSG:32615',photometric='RGB') as ds:ds.write(samples)
            image,info=read_map(p,value_range=(0,4095))
            np.testing.assert_array_equal(image[1,1],[128,128,128,255])
            self.assertEqual(image[11,11,3],0);self.assertEqual(image[20,20,3],255)

    def test_fourth_non_alpha_band_does_not_crop_map(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rgb_extra.tif';samples=np.full((4,32,32),80,np.uint8);samples[3]=0
            with rasterio.open(p,'w',driver='GTiff',width=32,height=32,count=4,dtype='uint8',photometric='RGB',
                               transform=Affine(.1,0,500000,0,-.1,4000000),crs='EPSG:32615') as ds:ds.write(samples)
            image,info=read_map(p)
            self.assertIsNone(info['alpha_band']);self.assertTrue(np.all(image[:,:,3]==255))


if __name__=='__main__':unittest.main(verbosity=2)
