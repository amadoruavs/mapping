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

    def validate(self):
        if self.altitude_agl_m <= 0 or not 0 < self.hfov_deg < 175: raise ValueError('Invalid altitude / horizontal FOV')
        if self.vfov_deg is not None and not 0 < self.vfov_deg < 175: raise ValueError('Invalid vertical FOV')
        if self.digital_zoom < 1 or self.tile_size < 1 or self.vocabulary_size < 3: raise ValueError('Invalid zoom / patch size')
        if self.max_output_dimension < 64 or self.registration_width < 64: raise ValueError('Image dimensions too small')
        if not 0 < self.sun_elevation_deg < 90: raise ValueError('Sun elevation must be between horizon and zenith')
        if self.geo_hash_cell_m <= 0 or self.local_hash_cell_px <= 0 or self.gps_tolerance_m <= 0: raise ValueError('Cell sizes and tolerance must be positive')
        if self.lighting_candidates<1 or self.shadow_steps<1 or self.selection_sigma_rgb<=0: raise ValueError('Invalid lighting / sampling settings')
        if not 0<=self.light_strength<=1: raise ValueError('Lighting strength must be in [0,1]')
        return self

    def to_dict(self): return asdict(self)
