# Hard lessons

Write these on the inside of your forehead before touching alignment or a README.

## 1. A tiny pixel residual can mean the map vanished

The 600-iteration 564-photo run reported median 0.0028 px. Most images had collapsed to a speck. **Always inspect footprint area** (`transforms.npy`, registration report). If median area ratio is ~1e-6, the objective is lying.

## 2. The visual objective must be in *source* pixels

Mapped-coordinate error rewards shrinking. Bidirectional transfer error in the original (resized) photo pixels does not. That is `alignment_symmetric_transfer`. Symmetric convergence also needs low damping (< 0.01); the float32 floor of the objective is about 1e-5.

## 3. GPS is a scale/north prior, not a survey

`--gps-centers` fits a similarity after the mosaic exists. Median residual ≤ 5 m is a **consistency gate**, not accuracy. SIYI files that contain coordinates still say `NO GPS connected`. Recorded gimbal yaw can be 180° off and was excluded from several successful fits.

## 4. `allow_gps_mismatch` does not fix the mosaic

It only skips the final 5 m median check. Use it when the user explicitly wants a map anyway. Write the residual into `geographic_alignment.json` either way.

## 5. Do not add GPS/yaw into the *inner* solve unless asked

A geo-priors experiment was written, then the user rolled the tree back to `runtime.zip`. The backup is in `backups/`. Do not resurrect it as a drive-by improvement.

## 6. Disconnected graphs are a hard error

Never drop frames to make a connected component. Recovery may retry verified visual pairs at 1280. It must not create GPS-only edges.

## 7. Generated FSR frames can invent pavement

Frame interpolation on nadir mosaics produced duplicated/warped roads. ISM default is interpolation off. If you re-enable it, keep `fsr_geometry.py` gates and the "both parents agree" validation kernel.

## 8. One source per pixel

Averaging two photos at a seam looks like ghosting. Coherent labels plus a *slow* exposure field looked better. Parallax on buildings/trees will still cut.

## 9. Presentation is not geometry

Sharpen, CLAHE, grade, OpenRT, FSR2×, lasso — none of these repair a bad homography. Time them separately from mapping. Do not quote a 9 s grade replay as an end-to-end mapping time.

## 10. Byte-identical is a gift. FSR rasters are not

LeanADM maps were pixel-identical across the lean trim. FSR-generated / FSR-upscaled rasters were **not** bit-identical across separate runs (example mean BGRA abs diff 0.051/255). Do not claim exact FSR repeatability.

## 11. Different residual definitions are not rankable

- Mapped-coordinate soft-L1
- Symmetric source-pixel transfer
- Image-center to GPS (near-nadir proxy)
- Physical camera-center to GPS
- ODM 3D reprojection
- Held-out checkpoint RMSE

Pick one, name it, and do not "improve" 2.38 m by quoting 0.0028 px.

## 12. Docker ODM and native Metal are different computers

12-CPU ARM64 Docker with 7.75 GiB vs native M4 Pro Metal. Stage mix is different. Environment is different. Speedup headlines are almost always misleading.

## 13. Empty output directories

Every launcher refuses to clobber. If you "just rerun" into `final_test`, you will fail or the user will lose a map. Make a new folder.

## 14. AGL is configured

EXIF altitude is usually ellipsoid/sea-level. The SIYI flights used 51.816 m or 60 m as a human number. Wrong AGL wrongs neighbor search and GSD, not just a label.

## 15. Every-sixth ODM is not WebODM quality

Dedup + stride 6 was an explicit product decision for the Fedora app. Do not market it as default ODM quality.

## 16. Host-only AMD kernels

`acceleration/kernels` were written for HIP/XDNA2. They were compiled/tested as host C++ math on a Mac. `hardware_execution` is a specific certification flag. Do not set it from a host test.

## 17. Memory guards fire on real flights

564 photos × 1280 registration hit a 512 MiB alignment scratch cap. 1024 MiB is the ISM default now. Planned scratch grows with `(8N)²`.

## 18. The user will delete the machine copy

Anything not in this git repo is gone. Do not promise to "look at last week's `output/killerofmaps` PNG." It was not uploaded.
