"""Edit flight assumptions here. Altitude is ABOVE GROUND, never EXIF sea level."""
from dataclasses import dataclass, asdict

ALTITUDE_AGL_M = 30.0  # PLACEHOLDER: set to the actual flight height before geographic use.
HORIZONTAL_FOV_DEG = 81.0
VERTICAL_FOV_DEG = None  # Derive from actual aspect ratio and square pixels.
DEFAULT_PITCH_DEG = -90.0  # Nadir; override for tilted gimbal imagery.
DEFAULT_ROLL_DEG = 0.0

CAMERA_SPEC = {
    'model': 'SIYI A8 Mini', 'sensor': 'Sony 1/1.7-inch optical format',
    'effective_megapixels': 8, 'equivalent_focal_length_mm': 21, 'f_number': 2.8,
    'published_hfov_deg': 81, 'published_dfov_deg': 93, 'published_vfov_deg': 65,
    'physical_sensor_dimensions_mm': None, 'shutter': 'rolling',
    'sources': ['https://www.siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/spec/',
                'https://www.siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/faq/'],
    'note': 'Optical format is not a physical diagonal. Published FOVs do not imply the same crop at 16:9. No unverified sensor dimensions or distortion coefficients are assumed.'
}

@dataclass
class Config:
    altitude_agl_m: float = ALTITUDE_AGL_M
    hfov_deg: float = HORIZONTAL_FOV_DEG
    vfov_deg: float | None = VERTICAL_FOV_DEG
    default_pitch_deg: float = DEFAULT_PITCH_DEG
    default_roll_deg: float = DEFAULT_ROLL_DEG
    digital_zoom: float = 1.0
    tile_size: int = 21
    vocabulary_size: int = 50
    geo_hash_cell_m: float = .5
    local_hash_cell_px: float = 21.0
    gps_tolerance_m: float = 5.0
    allow_gps_mismatch: bool = False  # Continue after the GPS-center residual check fails.
    delta_rmse_threshold: float = .02
    corner_min_ncc: float = .8
    corner_min_std: float = 3.0
    template_search_radius: int = 24
    visual_fallback: bool = True
    skip_first: int = 7
    near_duplicate_mae: float = 1.5
    registration_width: int = 1400
    feature_count: int = 7000
    sequence_neighbors: int = 6
    max_output_dimension: int = 6500
    seed: int = 2026
    selection_sigma_rgb: float = 22.0
    sun_elevation_deg: float = 30.0
    sun_azimuth_deg: float = 255.0
    light_strength: float = .22
    lighting_candidates: int = 5
    shadow_max_distance_m: float = 100.0
    shadow_steps: int = 48
    metal_recompute_warps: bool = False
    gpu_alignment: bool = False
    alignment_budget_mb: int = 512
    alignment_max_iterations: int = 60  # GPU solver cap; convergence can stop earlier
    alignment_symmetric_transfer: bool = False  # measure matches in both source images
    geographic_solver: str = 'visual'  # visual or opensfm_planar (joint physical camera poses)
    geographic_gps_sigma_m: float = 10.0  # ODM fallback when measured accuracy is absent
    geographic_altitude_sigma_m: float = 10.0  # configured AGL uncertainty; not EXIF altitude
    geographic_orientation_sigma_deg: float = 10.0  # uncertainty of the recorded camera axes
    coherent_composite: bool = False
    coherent_budget_mb: int = 32  # coarse observations + unary/confidence for source selection
    compact_outputs: bool = False
    stylized_shadows: bool = False
    lasso_crop: bool = False
    lasso_inset_ratio: float = .005
    lasso_opening_ratio: float = .06
    lasso_angular_radius: int = 12
    export_profile: str = 'diagnostic'
    fsr: bool = False  # spatial EASU during rendering, then RCAS; requires coherent Metal
    fsr_strength: float = .5
    input_upscale_passes: int = 0  # experimental preprocessing before SIFT and rendering
    render_upscale_passes: int | None = None  # None follows input; explicit value affects rendering only
    output_sharpen_passes: int = 1
    color_grade: str = 'none'
    regional_grade: str = 'none'
    interpolation_levels: int = 0  # experimental; render-only, after real-photo alignment
    presentation: bool = False  # user sharpening / Lanczos -> final color grade
    presentation_fsr_passes: int = 0  # optional historical ablation; off in current pipeline
    presentation_scale: float = 2.0
    presentation_raytrace: bool = False
    presentation_profile: str = 'standard'
    fsr_frame_budget_mb: int = 512
    grade_white_reduction: float = 0.0
    grade_vibrance_reduction: float = 0.0

    def validate(self):
        if type(self.allow_gps_mismatch) is not bool:
            raise ValueError('allow_gps_mismatch must be a boolean')
        if type(self.alignment_symmetric_transfer) is not bool:
            raise ValueError('alignment_symmetric_transfer must be a boolean')
        if self.alignment_symmetric_transfer and not self.gpu_alignment:
            raise ValueError('Symmetric transfer alignment requires the Metal solver')
        if type(self.alignment_max_iterations) is not int or not 1<=self.alignment_max_iterations<=600:
            raise ValueError('Alignment iteration limit must be an integer from 1 to 600')
        if type(self.alignment_budget_mb) is not int or not 128<=self.alignment_budget_mb<=2048:
            raise ValueError('Alignment workspace budget must be 128 to 2048 MiB')
        import math
        if not math.isfinite(self.lasso_inset_ratio) or not 0<=self.lasso_inset_ratio<=.1:
            raise ValueError('Lasso inset must be between zero and .1')
        if not math.isfinite(self.lasso_opening_ratio) or not 0<=self.lasso_opening_ratio<=.15:
            raise ValueError('Lasso opening ratio must be between zero and .15')
        if type(self.lasso_angular_radius) is not int or not 0<=self.lasso_angular_radius<=64:
            raise ValueError('Lasso angular radius must be an integer from zero to 64')
        if self.geographic_solver not in ('visual','opensfm_planar'):
            raise ValueError('Unknown geographic solver')
        if self.geographic_solver=='opensfm_planar' and not self.gpu_alignment:
            raise ValueError('OpenSfM planar convergence requires Metal/MPS')
        for name in ('geographic_gps_sigma_m','geographic_altitude_sigma_m','geographic_orientation_sigma_deg'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be finite and positive')
        from .cinematic_grade import PRESETS,REGION_PRESETS
        if self.presentation_profile not in ('standard','siyi_low_light'):raise ValueError('Unknown presentation profile')
        if type(self.fsr_frame_budget_mb) is not int or not 128<=self.fsr_frame_budget_mb<=1024:raise ValueError('FSR frame budget must be 128..1024 MiB')
        if type(self.presentation_raytrace) is not bool:raise ValueError('presentation_raytrace must be a boolean')
        if self.presentation_raytrace and not self.presentation:raise ValueError('Ray tracing requires the Metal presentation pipeline')
        if not math.isfinite(self.grade_white_reduction) or not 0<=self.grade_white_reduction<.5:raise ValueError('White reduction must be from 0 to less than .5')
        if not math.isfinite(self.grade_vibrance_reduction) or not 0<=self.grade_vibrance_reduction<=1:raise ValueError('Vibrance reduction must be from 0 to 1')
        if self.grade_white_reduction or self.grade_vibrance_reduction:
            raise ValueError('The supplied grader replaces legacy white/vibrance reductions; edit whites/saturation in its preset')
        if type(self.presentation_fsr_passes) is not int or not 0<=self.presentation_fsr_passes<=2:raise ValueError('Presentation FSR passes must be 0, 1 or 2')
        if not math.isfinite(self.presentation_scale) or not 1<=self.presentation_scale<=4:raise ValueError('Presentation Lanczos scale must be 1 to 4')
        if self.presentation and (not self.coherent_composite or self.export_profile!='png'):raise ValueError('Presentation requires coherent Metal PNG output')
        if self.color_grade!='none' and self.color_grade not in PRESETS:raise ValueError('Unknown color grade')
        if self.regional_grade!='none' and self.regional_grade not in REGION_PRESETS:raise ValueError('Unknown regional grade')
        if self.regional_grade!='none' and self.color_grade=='none':raise ValueError('Regional grading requires a global preset')
        if type(self.interpolation_levels) is not int or not 0<=self.interpolation_levels<=2:raise ValueError('Interpolation levels must be 0, 1 or 2')
        if self.interpolation_levels and (not self.coherent_composite or self.export_profile!='png'):
            raise ValueError('Experimental interpolation requires coherent Metal and PNG output')
        if not 0 <= self.fsr_strength <= 1: raise ValueError('FSR strength must be in [0,1]')
        if type(self.input_upscale_passes) is not int or not 0 <= self.input_upscale_passes <= 3:
            raise ValueError('Input upscale passes must be an integer from 0 to 3')
        if self.render_upscale_passes is not None and (type(self.render_upscale_passes) is not int or not 0<=self.render_upscale_passes<=3):
            raise ValueError('Render upscale passes must be None or an integer from 0 to 3')
        if type(self.output_sharpen_passes) is not int or not 1 <= self.output_sharpen_passes <= 4:
            raise ValueError('Output sharpen passes must be an integer from 1 to 4')
        if (self.input_upscale_passes or self.render_upscale_passes) and not self.fsr: raise ValueError('Input/render upscaling requires FSR')
        if self.fsr and not self.coherent_composite: raise ValueError('FSR requires coherent Metal compositing')
        if self.export_profile not in ('vectors','png','diagnostic'):
            raise ValueError('Unknown export profile')
        if self.altitude_agl_m <= 0 or not 0 < self.hfov_deg < 175: raise ValueError('Invalid altitude / horizontal FOV')
        if self.vfov_deg is not None and not 0 < self.vfov_deg < 175: raise ValueError('Invalid vertical FOV')
        if self.digital_zoom < 1 or self.tile_size < 1 or self.vocabulary_size < 3: raise ValueError('Invalid zoom / patch size')
        if self.max_output_dimension < 64 or self.registration_width < 64: raise ValueError('Image dimensions too small')
        if not 0 < self.sun_elevation_deg < 90: raise ValueError('Sun elevation must be between horizon and zenith')
        if self.geo_hash_cell_m <= 0 or self.local_hash_cell_px <= 0 or self.gps_tolerance_m <= 0: raise ValueError('Cell sizes and tolerance must be positive')
        if self.lighting_candidates<1 or self.shadow_steps<1 or self.selection_sigma_rgb<=0: raise ValueError('Invalid lighting / sampling settings')
        if self.coherent_budget_mb<1: raise ValueError('Coherent composite budget must be at least 1 MiB')
        if not 0<=self.light_strength<=1: raise ValueError('Lighting strength must be in [0,1]')
        return self

    def to_dict(self): return asdict(self)
