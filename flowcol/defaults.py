"""Central place for FlowCol default settings."""

# Canvas / resolution
DEFAULT_CANVAS_RESOLUTION: tuple[int, int] = (1024, 1024)
RENDER_RESOLUTION_CHOICES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SUPERSAMPLE_CHOICES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

# LIC render settings
DEFAULT_RENDER_PASSES: int = 2
DEFAULT_STREAMLENGTH_FACTOR: float = 60.0 / 1024.0
DEFAULT_PADDING_MARGIN: float = 0.10
DEFAULT_NOISE_SEED: int = 0
DEFAULT_NOISE_SIGMA: float =0.0

# bryLIC tiling defaults (for performance)
DEFAULT_TILE_SHAPE: tuple[int, int] | None = (512, 512)  # None disables tiling
DEFAULT_NUM_THREADS: int | None = None  # None = auto-detect from CPU count
DEFAULT_EDGE_GAIN_STRENGTH: float = 0.0  # Disabled for now, creates halos around boundaries
DEFAULT_EDGE_GAIN_POWER: float = 2.0  # Falloff curve for edge gain effect

# Post-processing defaults
DEFAULT_HIGHPASS_SIGMA_FACTOR: float = 3.0 / 1024.0

# Downsampling defaults
DEFAULT_DOWNSAMPLE_SIGMA: float = 0.6

# Display postprocessing defaults
DEFAULT_CLIP_PERCENT: float = 0.5
MAX_CLIP_PERCENT: float = 1.5
DEFAULT_BRIGHTNESS: float = 0.0
DEFAULT_CONTRAST: float = 1.0
DEFAULT_GAMMA: float = 1.0

# Colorization defaults
DEFAULT_COLOR_ENABLED: bool = False
DEFAULT_COLOR_PALETTE: str = "Ink & Gold"

# UI interaction defaults
SCROLL_SCALE_SENSITIVITY: float = 0.02
