# Sample results

The final deliverable is [output/full_pipeline/map_golden.png](output/full_pipeline/map_golden.png), 5113 × 6500 pixels with transparent uncovered areas. The neutral comparison is [output/full_pipeline/map_neutral.png](output/full_pipeline/map_neutral.png).

46 supplied JPEGs → 27 retained frames: 7 initial/takeoff frames, 11 annotated frames, and 1 frame containing an aircraft part were excluded. No exact or near-identical clean frames met the conservative duplicate threshold. All 46 JPEGs lack EXIF and XMP; the sample therefore uses local map coordinates.

The complete pipeline indexed 508,923 RGB tiles of 21×21 pixels, including explicitly marked partial edge tiles, and saved 108 full-resolution corner vocabularies. A spatial multimap retains all overlapping observations.

The strict corner experiment accepted **0 of 189 evaluated patches** at the 2% delta-RMSE threshold. The best patch was about 4.17%; median was about 11.39%. Thus this dataset does not validate the corner-only proposal. The completed visual map used 120 explicit SIFT/RANSAC fallback edges from 141 sequential-neighbor candidate pairs.

Joint alignment converged. Median correspondence residual was 0.879 pixels and the 95th percentile was 2.787 pixels at 1,400-pixel working width. These are internal image-consistency measurements, not independent survey accuracy.

The first complete 27-frame pass took about 133 seconds on this machine. The final pass reused pair matches and took 102.7 seconds. No 300–450-frame or Radeon-integrated-GPU benchmark has been run. Full-size intermediate observations remain on disk for inspection.

Golden-hour PNG output uses a subtle linear-light SDR treatment. No height map was available, so it does not invent cast shadows or terrain. Linear .hdr/.npy exports preserve the shader output range but cannot recover clipped scene radiance from JPEGs. The optional height-map branch implements normals and per-environment shadow sampling.

Validation: 10 tests passed, including actual embedded GPS EXIF, XMP overrides, yaw/ground geometry, multivalue hash storage, photometric delta matching, shadow direction, duplicate/annotation filtering, height-lighting export, and the entire geographic pipeline on known synthetic overlap. The synthetic geographic run accepted 3 corner patches, demonstrating the positive verification path. The actual Metal lighting kernel matched the CPU shader to a maximum absolute error of 1.79e-07 across 1,457 randomized pixels.

Next input: original GPS/camera-heading metadata and the real flight altitude. Update adm/metadata.py for the actual EXIF/flight-log fields and adm/config.py for calibration. Altitude 30 m remains a placeholder; the source frames visibly vary in viewpoint, so a hardcoded constant may need to become per-frame altitude.
