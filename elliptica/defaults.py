"""Central place for Elliptica default settings."""

# Canvas / resolution
DEFAULT_CANVAS_RESOLUTION: tuple[int, int] = (1024, 1024)
RENDER_RESOLUTION_CHOICES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SUPERSAMPLE_CHOICES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

# LIC render settings
DEFAULT_RENDER_PASSES: int = 2
DEFAULT_STREAMLENGTH_FACTOR: float = 60.0 / 1024.0
DEFAULT_PADDING_MARGIN: float = 0.10
DEFAULT_NOISE_SEED: int = 0
DEFAULT_NOISE_SIGMA: float =0.5
DEFAULT_USE_MASK: bool = True  # Block streamlines at boundaries
DEFAULT_SOLVE_SCALE: float = 1.0  # PDE solve resolution relative to render grid
MIN_SOLVE_SCALE: float = 0.1
MAX_SOLVE_SCALE: float = 1.0
SOLVE_RELAX_BAND: int = 3  # Relaxation band width when solve_scale < 1
SOLVE_RELAX_ITERS: int = 8
SOLVE_RELAX_OMEGA: float = 0.8

# bryLIC tiling defaults (for performance)
DEFAULT_TILE_SHAPE: tuple[int, int] | None = (512, 512)  # None disables tiling
DEFAULT_NUM_THREADS: int | None = None  # None = auto-detect from CPU count
DEFAULT_EDGE_GAIN_STRENGTH: float = 0.5  # Disabled for now, creates halos around boundaries
DEFAULT_EDGE_GAIN_POWER: float = 2.0  # Falloff curve for edge gain effect
MIN_EDGE_GAIN_STRENGTH: float = -3.0
MAX_EDGE_GAIN_STRENGTH: float = 3.0
MIN_EDGE_GAIN_POWER: float = 0.1
MAX_EDGE_GAIN_POWER: float = 3.0

# Post-processing defaults
DEFAULT_HIGHPASS_SIGMA_FACTOR: float = 3.0 / 1024.0

# Downsampling defaults
DEFAULT_DOWNSAMPLE_SIGMA: float = 0.6
MAX_DOWNSAMPLE_SIGMA: float = 2.0

# Display postprocessing defaults
DEFAULT_CLIP_PERCENT: float = 0.5
MAX_CLIP_PERCENT: float = 2.0
DEFAULT_BRIGHTNESS: float = 0.0
MIN_BRIGHTNESS: float = -0.9
MAX_BRIGHTNESS: float = 0.9
DEFAULT_CONTRAST: float = 1.0
MIN_CONTRAST: float = 0.5
MAX_CONTRAST: float = 2.0
DEFAULT_GAMMA: float = 1.0
MIN_GAMMA: float = 0.3
MAX_GAMMA: float = 3.0

# Colorization defaults
DEFAULT_COLOR_ENABLED: bool = True
DEFAULT_COLOR_PALETTE: str = "Ink Wash"

# Smear defaults (stored as fraction of canvas width for resolution independence)
# Example: 0.002 = 0.2% of canvas width (2px at 1k, 14px at 7k)
DEFAULT_SMEAR_SIGMA: float = 0.0005  # 0.2% of canvas width
MIN_SMEAR_SIGMA: float = 0.00001     # 0.01% (0.1px at 1k, 0.7px at 7k)
MAX_SMEAR_SIGMA: float = 0.005       # 0.5% (5px at 1k, 35px at 7k)

# UI interaction defaults
SCROLL_SCALE_SENSITIVITY: float = 0.02
