"""Central place for FlowCol default settings."""

# Canvas / resolution
DEFAULT_CANVAS_RESOLUTION: tuple[int, int] = (1024, 1024)
RENDER_RESOLUTION_CHOICES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SUPERSAMPLE_CHOICES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

# LIC render settings
DEFAULT_RENDER_PASSES: int = 2
DEFAULT_STREAMLENGTH_FACTOR: float = 60.0 / 1024.0
DEFAULT_PADDING_MARGIN: float = 0.10
DEFAULT_NOISE_SEED: int = 0
DEFAULT_NOISE_SIGMA: float =0.0

# Post-processing defaults
DEFAULT_HIGHPASS_SIGMA_FACTOR: float = 3.0 / 1024.0
DEFAULT_CLAHE_CLIP_LIMIT: float = 0.01
DEFAULT_CLAHE_KERNEL_ROWS: int = 8
DEFAULT_CLAHE_KERNEL_COLS: int = 8
DEFAULT_CLAHE_BINS: int = 150
DEFAULT_CLAHE_STRENGTH: float = 1.0

# Downsampling defaults
DEFAULT_DOWNSAMPLE_SIGMA: float = 0.6

# Display postprocessing defaults
DEFAULT_CLIP_PERCENT: float = 0.5
MAX_CLIP_PERCENT: float = 1.5
DEFAULT_CONTRAST: float = 1.0
DEFAULT_GAMMA: float = 1.0
