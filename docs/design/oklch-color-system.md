# OKLCH Color System Design Document

## Overview

This document describes the architecture for a flexible OKLCH-based color system that maps multiple physical fields to perceptually uniform color channels. The system replaces the current single-field → 1D LUT approach with a powerful three-channel composition pipeline.

### Current State

```
LIC_scalar → normalize → palette_LUT → RGB
```

Single scalar field, single artistic dimension (which palette).

### Target State

```
{LIC, Ex, Ey, φ, |E|, θ, ...} → user_expressions → (L, C, H) → gamut_map → RGB
```

Multiple fields, three independent artistic dimensions, full expression language.

**Key insight**: The available fields depend on which PDE is being solved. Electrostatics has `ex`, `ey`, `phi`. Fluid dynamics would have `vx`, `vy`, `pressure`, `vorticity`. The color system must be agnostic to the specific PDE — it receives whatever fields the solver provides.

---

## Motivation

The current palette-based colorization is artistically limiting:

1. **Single input** — Only LIC intensity drives color
2. **Coupled channels** — Can't independently control luminance vs. hue
3. **No directional encoding** — Field direction information is lost
4. **Limited exploration** — Finding good palettes is trial-and-error

The OKLCH system enables:

- **Direction in hue** — Electric field angle → rainbow encoding
- **Magnitude in luminance** — Strong fields = bright
- **LIC texture preserved** — Fine detail in any channel
- **Independent control** — Adjust saturation without affecting brightness
- **Perceptual uniformity** — OKLCH is designed for human perception

---

## Why OKLCH?

OKLCH (Oklab Lightness-Chroma-Hue) is a modern perceptually uniform color space:

| Property | Benefit |
|----------|---------|
| **Perceptual uniformity** | Equal numerical steps = equal perceived steps |
| **Hue linearity** | Hue interpolation doesn't pass through mud |
| **Lightness independence** | Can adjust L without shifting perceived hue |
| **Chroma predictability** | Saturation behaves intuitively |

Alternatives considered:

- **HSL/HSV** — Not perceptually uniform, hue shifts under saturation changes
- **Lab/LCH** — Better than HSL, but Oklab improves on blue-yellow axis
- **RGB direct** — No perceptual meaning, poor for artistic control

Reference: https://bottosson.github.io/posts/oklab/

---

## System Architecture

### Conceptual Layers

```
┌─────────────────────────────────────────────────────────────┐
│  UI Layer                                                    │
│  Preset selector, expression editors, channel controls       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Composition Pipeline                                        │
│  Field registry, channel evaluation, OKLCH assembly          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  Expression System       │    │  Color Space Math        │
│  AST, parser, compiler   │    │  OKLCH ↔ sRGB, gamut     │
└──────────────────────────┘    └──────────────────────────┘
```

### Module Structure

```
flowcol/
├── colorspace/                    # Pure color math (no flowcol deps)
│   ├── __init__.py
│   ├── oklch.py                  # OKLCH ↔ OKLab ↔ Linear RGB ↔ sRGB
│   ├── gamut.py                  # Gamut compression/clipping
│   └── tests/
│       ├── test_oklch.py
│       └── test_gamut.py
│
├── expr/                          # Expression system (generic, reusable)
│   ├── __init__.py
│   ├── ast.py                    # Expression AST nodes
│   ├── parser.py                 # String → AST
│   ├── compiler.py               # AST → callable (numpy/torch)
│   ├── functions.py              # Built-in function registry
│   ├── errors.py                 # ParseError, EvalError, etc.
│   └── tests/
│       ├── test_parser.py
│       └── test_compiler.py
│
├── color/                         # OKLCH composition pipeline
│   ├── __init__.py
│   ├── fields.py                 # FieldRegistry, FieldContext
│   ├── channel.py                # ChannelConfig
│   ├── compose.py                # OKLCHConfig, compose_oklch()
│   ├── presets.py                # Built-in artistic presets
│   ├── config.py                 # ColorConfig (unified)
│   ├── palette.py                # Legacy palette code (from render.py)
│   ├── pipeline.py               # build_rgb() entry point
│   └── tests/
│
├── gpu/
│   ├── ...existing...
│   └── colorspace.py             # GPU OKLCH operations
```

### Design Principles

1. **Separation of concerns** — Color math, expressions, and composition are independent
2. **Backend agnostic** — Same logic for numpy (CPU) and torch (GPU)
3. **Lazy computation** — Derived fields computed only when needed
4. **Safe expressions** — Whitelist-only, no arbitrary code execution
5. **Backward compatible** — Palette mode unchanged, OKLCH is opt-in
6. **PDE agnostic** — Field definitions come from solvers, not hardcoded

---

## PDE-Agnostic Architecture

A key design goal is that the color system works with any PDE solver, not just electrostatics.

### The Problem

Different PDEs produce different physical quantities:

| PDE | Vector Field | Scalar Fields | Natural Derived Fields |
|-----|--------------|---------------|------------------------|
| Electrostatics | E = (Ex, Ey) | φ (potential) | \|E\|, θ_E |
| Fluid dynamics | v = (vx, vy) | p (pressure) | \|v\|, θ_v, ω (vorticity) |
| Heat equation | q = (qx, qy) | T (temperature) | \|q\|, θ_q |
| Magnetostatics | B = (Bx, By) | A (potential) | \|B\|, θ_B |

Hardcoding field names like `ex`, `ey` would make the color system useless for other PDEs.

### The Solution

**Solvers provide field metadata via a `FieldProvider` protocol:**

```python
class FieldProvider(Protocol):
    def primary_field_info(self) -> dict[str, FieldInfo]: ...
    def derived_field_defs(self) -> dict[str, DerivedFieldDef]: ...
```

**The color system is field-agnostic:**
- `FieldContext` accepts any fields the solver provides
- Expressions reference field names as variables
- Presets are organized by PDE type
- UI dynamically shows available fields

**Common patterns are reusable:**

```python
# Any PDE with a 2D vector field can use these:
make_magnitude_field('vx', 'vy', 'speed', ...)
make_angle_field('vx', 'vy', 'v_angle', ...)
```

### Data Flow with Multiple PDEs

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Electrostatics  │     │ Fluid Solver    │     │ Heat Solver     │
│ Solver          │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    FieldProvider           FieldProvider           FieldProvider
    (ex, ey, phi)           (vx, vy, p)             (T, qx, qy)
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  FieldContext          │
                    │  (PDE-agnostic)        │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Expression Evaluation │
                    │  (field-agnostic)      │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  OKLCH → RGB           │
                    │  (color-space math)    │
                    └────────────────────────┘
```

### UI Implications

The UI must:
1. Query available fields from current solver
2. Show only compatible presets (or warn about incompatible ones)
3. Validate expressions against available fields
4. Display field metadata (description, range, unit) in expression help

---

## Module Specifications

### 1. colorspace/ — Pure Color Mathematics

Zero dependencies on flowcol. Could be extracted as standalone library.

#### oklch.py

```python
"""OKLCH color space conversions.

Reference: https://bottosson.github.io/posts/oklab/

All functions accept scalars, numpy arrays, or torch tensors.
GPU acceleration automatic when torch GPU tensors are passed.
"""

# === Core Conversions ===

def oklch_to_oklab(L, C, H):
    """OKLCH → OKLab. H in degrees."""
    H_rad = H * (pi / 180)
    a = C * cos(H_rad)
    b = C * sin(H_rad)
    return L, a, b

def oklab_to_oklch(L, a, b):
    """OKLab → OKLCH. Returns H in degrees [0, 360)."""
    C = sqrt(a**2 + b**2)
    H = atan2(b, a) * (180 / pi) % 360
    return L, C, H

def oklab_to_linear_rgb(L, a, b):
    """OKLab → Linear RGB via LMS intermediate."""
    # OKLab → LMS (cube root space)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Cube to get LMS
    l, m, s = l_**3, m_**3, s_**3

    # LMS → Linear RGB (matrix multiply)
    r = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
    g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
    b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s

    return r, g, b

def linear_rgb_to_oklab(r, g, b):
    """Linear RGB → OKLab via LMS intermediate."""
    # Linear RGB → LMS
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b

    # Cube root
    l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)

    # LMS cube root → OKLab
    L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_

    return L, a, b

def linear_to_srgb(rgb):
    """Linear RGB → sRGB gamma encoding."""
    # Piecewise: linear below threshold, power curve above
    threshold = 0.0031308
    low = rgb * 12.92
    high = 1.055 * pow(rgb, 1/2.4) - 0.055
    return where(rgb <= threshold, low, high)

def srgb_to_linear(rgb):
    """sRGB → Linear RGB gamma decoding."""
    threshold = 0.04045
    low = rgb / 12.92
    high = pow((rgb + 0.055) / 1.055, 2.4)
    return where(rgb <= threshold, low, high)

# === Convenience Composites ===

def oklch_to_srgb(L, C, H):
    """OKLCH → sRGB in one call. Returns (H, W, 3) array."""
    L, a, b = oklch_to_oklab(L, C, H)
    r, g, b = oklab_to_linear_rgb(L, a, b)
    rgb_linear = stack([r, g, b], axis=-1)
    return linear_to_srgb(rgb_linear)

def srgb_to_oklch(rgb):
    """sRGB → OKLCH. Input (H, W, 3), returns (L, C, H) tuple."""
    rgb_linear = srgb_to_linear(rgb)
    r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
    L, a, b = linear_rgb_to_oklab(r, g, b)
    return oklab_to_oklch(L, a, b)
```

#### gamut.py

```python
"""Gamut mapping for out-of-gamut OKLCH values.

Not all (L, C, H) combinations produce valid sRGB. High chroma at
extreme lightness is particularly problematic.

Strategies:
- clip: Hard-clip RGB to [0,1] — fast but can flatten regions
- compress: Reduce C until in-gamut — preserves L and H intent
"""

def is_in_gamut(L, C, H, tolerance=1e-4):
    """Check if OKLCH values produce valid sRGB (all channels in [0,1])."""
    rgb = oklch_to_srgb(L, C, H)
    return (rgb >= -tolerance).all(axis=-1) & (rgb <= 1 + tolerance).all(axis=-1)

def max_chroma_for_lh(L, H, steps=16):
    """Find maximum valid chroma for given L and H via binary search.

    This is the expensive but accurate approach. For real-time use,
    consider precomputed lookup tables.
    """
    lo = zeros_like(L)
    hi = full_like(L, 0.5)  # 0.5 is always out of gamut

    for _ in range(steps):
        mid = (lo + hi) / 2
        valid = is_in_gamut(L, mid, H)
        lo = where(valid, mid, lo)
        hi = where(~valid, mid, hi)

    return lo

def gamut_compress(L, C, H, method='chroma'):
    """Bring out-of-gamut colors into sRGB gamut.

    Methods:
    - 'chroma': Reduce C toward 0 until in-gamut (preserves L, H)
    - 'clip': Just clip RGB (fast, may distort)
    """
    if method == 'clip':
        rgb = oklch_to_srgb(L, C, H)
        rgb_clipped = clip(rgb, 0, 1)
        return srgb_to_oklch(rgb_clipped)  # back to OKLCH for consistency

    elif method == 'chroma':
        max_C = max_chroma_for_lh(L, H)
        C_compressed = minimum(C, max_C)
        return L, C_compressed, H

def gamut_clip_rgb(L, C, H):
    """Convert to RGB and hard-clip. Returns RGB directly."""
    rgb = oklch_to_srgb(L, C, H)
    return clip(rgb, 0, 1)
```

**Gamut Mapping Considerations:**

The binary search approach in `max_chroma_for_lh` is accurate but expensive. Options for optimization:

1. **Precomputed LUT** — 256×360 table of max chroma for (L, H) pairs, ~360KB
2. **Analytical approximation** — Björn Ottosson has published closed-form approximations
3. **Adaptive precision** — Fewer binary search steps for interactive preview, more for export

For interactive use, recommend: coarse LUT + linear interpolation.

---

### 2. expr/ — Expression System

Generic expression parsing and evaluation. No flowcol dependencies.

#### ast.py

```python
"""Expression AST nodes.

The AST is immutable (frozen dataclasses) for safety and hashability.
"""

from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Const:
    """Numeric constant."""
    value: float

@dataclass(frozen=True)
class Var:
    """Variable reference (field name)."""
    name: str

@dataclass(frozen=True)
class BinOp:
    """Binary operation."""
    op: str        # '+', '-', '*', '/', '^'
    left: 'Expr'
    right: 'Expr'

@dataclass(frozen=True)
class UnaryOp:
    """Unary operation."""
    op: str        # '-' (negation only, functions handle the rest)
    arg: 'Expr'

@dataclass(frozen=True)
class FuncCall:
    """Function call."""
    name: str
    args: tuple['Expr', ...]

# Union type for all expression nodes
Expr = Union[Const, Var, BinOp, UnaryOp, FuncCall]


def variables_in(expr: Expr) -> set[str]:
    """Extract all variable names from an expression."""
    match expr:
        case Const(_):
            return set()
        case Var(name):
            return {name}
        case BinOp(_, left, right):
            return variables_in(left) | variables_in(right)
        case UnaryOp(_, arg):
            return variables_in(arg)
        case FuncCall(_, args):
            return set().union(*(variables_in(a) for a in args))


def pretty_print(expr: Expr) -> str:
    """Convert AST back to readable string."""
    match expr:
        case Const(v):
            return str(v)
        case Var(name):
            return name
        case BinOp(op, left, right):
            return f"({pretty_print(left)} {op} {pretty_print(right)})"
        case UnaryOp('-', arg):
            return f"-{pretty_print(arg)}"
        case FuncCall(name, args):
            args_str = ", ".join(pretty_print(a) for a in args)
            return f"{name}({args_str})"
```

#### parser.py

```python
"""Recursive descent parser for expressions.

Grammar (informal):
    expr     := term (('+' | '-') term)*
    term     := power (('*' | '/') power)*
    power    := unary ('^' power)?          # right-associative
    unary    := '-' unary | call
    call     := atom ('(' args? ')')?
    atom     := NUMBER | IDENT | '(' expr ')'
    args     := expr (',' expr)*

Examples:
    "0.5"
    "lic"
    "pow(lic, 0.8)"
    "e_mag * 0.5 + 0.25"
    "sqrt(ex^2 + ey^2)"
    "mod(e_angle + 90, 360)"
    "clamp(phi * 2 - 1, 0, 1)"
    "-sin(e_angle * rad(1))"
"""

import re
from .ast import Const, Var, BinOp, UnaryOp, FuncCall, Expr
from .errors import ParseError

# Token types
TOKEN_PATTERNS = [
    ('NUMBER', r'\d+\.?\d*|\.\d+'),    # 123, 1.5, .5
    ('IDENT',  r'[a-zA-Z_][a-zA-Z_0-9]*'),
    ('OP',     r'[+\-*/^]'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('COMMA',  r','),
    ('SKIP',   r'\s+'),
]

TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_PATTERNS))


@dataclass
class Token:
    type: str
    value: str
    pos: int


def tokenize(source: str) -> list[Token]:
    """Lexical analysis."""
    tokens = []
    for match in TOKEN_RE.finditer(source):
        type_ = match.lastgroup
        if type_ == 'SKIP':
            continue
        tokens.append(Token(type_, match.group(), match.start()))
    return tokens


class Parser:
    """Recursive descent parser."""

    def __init__(self, source: str):
        self.source = source
        self.tokens = tokenize(source)
        self.pos = 0

    def parse(self) -> Expr:
        expr = self.expr()
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            raise ParseError(f"Unexpected token: {tok.value}", tok.pos, self.source)
        return expr

    # ... recursive descent methods for each grammar rule ...


def parse(source: str) -> Expr:
    """Parse expression string to AST."""
    if not source.strip():
        raise ParseError("Empty expression", 0, source)
    return Parser(source).parse()
```

#### compiler.py

```python
"""Compile AST to executable functions.

Two backends:
- numpy: For CPU evaluation
- torch: For GPU evaluation (also works on CPU)

The "compilation" is really just building a closure that walks the AST.
For performance-critical cases, we could generate Python code and exec(),
but tree-walking is fast enough for our image sizes.
"""

import numpy as np
import torch
from .ast import Expr, Const, Var, BinOp, UnaryOp, FuncCall, variables_in
from .functions import FUNCTIONS
from .errors import UnknownVariableError, UnknownFunctionError, ArityError


class CompiledExpr:
    """Compiled expression ready for evaluation."""

    def __init__(self, ast: Expr):
        self.ast = ast
        self.variables = variables_in(ast)
        self._validate_functions()

    def _validate_functions(self):
        """Check all function calls reference known functions."""
        def check(node):
            match node:
                case FuncCall(name, args):
                    if name not in FUNCTIONS:
                        raise UnknownFunctionError(name, list(FUNCTIONS.keys()))
                    func = FUNCTIONS[name]
                    if not func.check_arity(len(args)):
                        raise ArityError(name, func.arity, len(args))
                    for arg in args:
                        check(arg)
                case BinOp(_, left, right):
                    check(left)
                    check(right)
                case UnaryOp(_, arg):
                    check(arg)
        check(self.ast)

    def validate_variables(self, available: set[str]) -> list[str]:
        """Check all variables are available. Returns list of missing."""
        return list(self.variables - available)

    def eval_numpy(self, context: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate with numpy arrays."""
        missing = self.validate_variables(set(context.keys()))
        if missing:
            raise UnknownVariableError(missing[0], list(context.keys()))
        return self._eval_node(self.ast, context, 'numpy')

    def eval_torch(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate with torch tensors."""
        missing = self.validate_variables(set(context.keys()))
        if missing:
            raise UnknownVariableError(missing[0], list(context.keys()))
        return self._eval_node(self.ast, context, 'torch')

    def _eval_node(self, node: Expr, ctx: dict, backend: str):
        """Recursively evaluate AST node."""
        match node:
            case Const(value):
                # Return scalar that broadcasts with arrays
                return value

            case Var(name):
                return ctx[name]

            case BinOp('+', left, right):
                return self._eval_node(left, ctx, backend) + self._eval_node(right, ctx, backend)
            case BinOp('-', left, right):
                return self._eval_node(left, ctx, backend) - self._eval_node(right, ctx, backend)
            case BinOp('*', left, right):
                return self._eval_node(left, ctx, backend) * self._eval_node(right, ctx, backend)
            case BinOp('/', left, right):
                return self._eval_node(left, ctx, backend) / self._eval_node(right, ctx, backend)
            case BinOp('^', left, right):
                l = self._eval_node(left, ctx, backend)
                r = self._eval_node(right, ctx, backend)
                return np.power(l, r) if backend == 'numpy' else torch.pow(l, r)

            case UnaryOp('-', arg):
                return -self._eval_node(arg, ctx, backend)

            case FuncCall(name, args):
                func = FUNCTIONS[name]
                evaluated_args = [self._eval_node(a, ctx, backend) for a in args]
                impl = func.numpy_impl if backend == 'numpy' else func.torch_impl
                return impl(*evaluated_args)


def compile(source: str) -> CompiledExpr:
    """Parse and compile expression in one step."""
    from .parser import parse
    ast = parse(source)
    return CompiledExpr(ast)
```

#### functions.py

```python
"""Built-in function registry.

Each function has numpy and torch implementations.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch


@dataclass
class FuncDef:
    name: str
    arity: int | tuple[int, int]   # exact count or (min, max)
    numpy_impl: Callable
    torch_impl: Callable
    description: str

    def check_arity(self, n: int) -> bool:
        if isinstance(self.arity, int):
            return n == self.arity
        min_a, max_a = self.arity
        return min_a <= n <= max_a


# Helper for numpy/torch compatibility
def _np_clamp(x, lo, hi):
    return np.clip(x, lo, hi)

def _np_lerp(a, b, t):
    return a + (b - a) * t

def _torch_mod(x, y):
    # torch.fmod preserves sign of dividend, numpy.mod is true modulo
    # Use remainder for numpy-like behavior
    return torch.remainder(x, y)


FUNCTIONS: dict[str, FuncDef] = {
    # === Unary Math ===
    'abs':   FuncDef('abs', 1, np.abs, torch.abs,
                     "Absolute value"),
    'sqrt':  FuncDef('sqrt', 1, np.sqrt, torch.sqrt,
                     "Square root"),
    'cbrt':  FuncDef('cbrt', 1, np.cbrt, lambda x: torch.sign(x) * torch.abs(x).pow(1/3),
                     "Cube root"),
    'exp':   FuncDef('exp', 1, np.exp, torch.exp,
                     "Exponential e^x"),
    'log':   FuncDef('log', 1, np.log, torch.log,
                     "Natural logarithm"),
    'log10': FuncDef('log10', 1, np.log10, torch.log10,
                     "Base-10 logarithm"),

    # === Trigonometric ===
    'sin':   FuncDef('sin', 1, np.sin, torch.sin,
                     "Sine (radians)"),
    'cos':   FuncDef('cos', 1, np.cos, torch.cos,
                     "Cosine (radians)"),
    'tan':   FuncDef('tan', 1, np.tan, torch.tan,
                     "Tangent (radians)"),
    'asin':  FuncDef('asin', 1, np.arcsin, torch.asin,
                     "Arcsine → radians"),
    'acos':  FuncDef('acos', 1, np.arccos, torch.acos,
                     "Arccosine → radians"),
    'atan':  FuncDef('atan', 1, np.arctan, torch.atan,
                     "Arctangent → radians"),

    # === Binary Math ===
    'pow':   FuncDef('pow', 2, np.power, torch.pow,
                     "Power: pow(x, y) = x^y"),
    'mod':   FuncDef('mod', 2, np.mod, _torch_mod,
                     "Modulo (true mathematical mod)"),
    'min':   FuncDef('min', 2, np.minimum, torch.minimum,
                     "Element-wise minimum"),
    'max':   FuncDef('max', 2, np.maximum, torch.maximum,
                     "Element-wise maximum"),
    'atan2': FuncDef('atan2', 2, np.arctan2, torch.atan2,
                     "Two-argument arctangent: atan2(y, x) → radians"),

    # === Ternary ===
    'clamp': FuncDef('clamp', 3, _np_clamp, torch.clamp,
                     "Clamp to range: clamp(x, lo, hi)"),
    'lerp':  FuncDef('lerp', 3, _np_lerp,
                     lambda a, b, t: torch.lerp(a, b, t) if isinstance(t, torch.Tensor) else a + (b-a)*t,
                     "Linear interpolation: lerp(a, b, t) = a + (b-a)*t"),

    # === Angle Utilities ===
    'deg':   FuncDef('deg', 1, np.degrees, torch.rad2deg,
                     "Radians to degrees"),
    'rad':   FuncDef('rad', 1, np.radians, torch.deg2rad,
                     "Degrees to radians"),

    # === Smoothstep / Easing ===
    'smoothstep': FuncDef('smoothstep', 3,
        lambda lo, hi, x: np.clip((x - lo) / (hi - lo), 0, 1) ** 2 * (3 - 2 * np.clip((x - lo) / (hi - lo), 0, 1)),
        lambda lo, hi, x: torch.clamp((x - lo) / (hi - lo), 0, 1) ** 2 * (3 - 2 * torch.clamp((x - lo) / (hi - lo), 0, 1)),
        "Smooth interpolation: smoothstep(lo, hi, x)"),

    # === Sign / Rounding ===
    'sign':  FuncDef('sign', 1, np.sign, torch.sign,
                     "Sign function: -1, 0, or 1"),
    'floor': FuncDef('floor', 1, np.floor, torch.floor,
                     "Floor (round down)"),
    'ceil':  FuncDef('ceil', 1, np.ceil, torch.ceil,
                     "Ceiling (round up)"),
    'round': FuncDef('round', 1, np.round, torch.round,
                     "Round to nearest integer"),
    'frac':  FuncDef('frac', 1, lambda x: x - np.floor(x), lambda x: x - torch.floor(x),
                     "Fractional part: frac(x) = x - floor(x)"),
}


def list_functions() -> list[str]:
    """List all available function names."""
    return sorted(FUNCTIONS.keys())

def get_function_help(name: str) -> str:
    """Get description for a function."""
    if name not in FUNCTIONS:
        return f"Unknown function: {name}"
    f = FUNCTIONS[name]
    arity = f.arity if isinstance(f.arity, int) else f"{f.arity[0]}-{f.arity[1]}"
    return f"{name}({arity} args): {f.description}"
```

#### errors.py

```python
"""Expression system errors.

All errors include enough context to produce helpful error messages.
"""

class ExprError(Exception):
    """Base class for expression errors."""
    pass


class ParseError(ExprError):
    """Syntax error in expression."""

    def __init__(self, message: str, position: int, source: str):
        self.message = message
        self.position = position
        self.source = source
        super().__init__(self._format())

    def _format(self) -> str:
        # Show the expression with a pointer to the error position
        pointer = ' ' * self.position + '^'
        return f"{self.message}\n  {self.source}\n  {pointer}"


class UnknownVariableError(ExprError):
    """Referenced undefined variable."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        msg = f"Unknown variable '{name}'"
        if available:
            msg += f". Available: {', '.join(sorted(available))}"
        super().__init__(msg)


class UnknownFunctionError(ExprError):
    """Called undefined function."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        msg = f"Unknown function '{name}'"
        if available:
            # Suggest similar names
            similar = [f for f in available if f.startswith(name[0])][:5]
            if similar:
                msg += f". Did you mean: {', '.join(similar)}?"
        super().__init__(msg)


class ArityError(ExprError):
    """Wrong number of arguments to function."""

    def __init__(self, name: str, expected: int | tuple, actual: int):
        self.name = name
        self.expected = expected
        self.actual = actual
        if isinstance(expected, int):
            exp_str = str(expected)
        else:
            exp_str = f"{expected[0]}-{expected[1]}"
        super().__init__(f"Function '{name}' expects {exp_str} arguments, got {actual}")
```

---

### 3. color/ — OKLCH Composition Pipeline

This is the flowcol-specific layer that ties everything together.

#### fields.py — Dynamic, PDE-Agnostic Field System

The field system must handle different PDEs with different physical quantities:

| PDE | Primary Fields | Derived Fields |
|-----|----------------|----------------|
| Electrostatics | `ex`, `ey`, `phi` | `e_mag`, `e_angle` |
| Fluid dynamics | `vx`, `vy`, `p` | `speed`, `v_angle`, `vorticity` |
| Heat equation | `T`, `qx`, `qy` | `q_mag`, `q_angle` |
| Magnetostatics | `Bx`, `By`, `A` | `B_mag`, `B_angle` |

The color system doesn't know about these specifics — it receives field metadata from the solver.

```python
"""Field registry and context management.

Fields are the raw data that can drive color channels:
- Primary fields: provided by solver (varies by PDE type)
- Derived fields: computed on-demand (defined by solver)
- LIC field: always available (computed from any vector field)

The FieldContext provides lazy evaluation and caching for derived fields.
Field definitions come from the PDE solver, NOT hardcoded here.
"""

from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
import torch


@dataclass
class FieldInfo:
    """Metadata about a field (provided by solver)."""
    name: str
    description: str
    typical_range: tuple[float, float]   # for UI hints, auto-normalization
    unit: str = ''                        # '', 'degrees', 'V/m', 'K', etc.
    is_cyclic: bool = False              # True for angles


@dataclass
class DerivedFieldDef:
    """Definition of a derived field (provided by solver)."""
    name: str
    description: str
    typical_range: tuple[float, float]
    unit: str
    is_cyclic: bool
    dependencies: tuple[str, ...]
    compute: Callable[[dict], np.ndarray | torch.Tensor]


class FieldProvider(Protocol):
    """Protocol that PDE solvers implement to provide field metadata."""

    def primary_field_info(self) -> dict[str, FieldInfo]:
        """Metadata for primary fields this solver produces."""
        ...

    def derived_field_defs(self) -> dict[str, DerivedFieldDef]:
        """Definitions for derived fields (computed from primaries)."""
        ...


# === Common Derived Field Computations ===
# These are reusable building blocks for solvers to use

def make_magnitude_field(x_name: str, y_name: str, out_name: str, description: str, unit: str):
    """Create a magnitude derived field from two components."""
    def compute(ctx):
        x, y = ctx[x_name], ctx[y_name]
        if isinstance(x, torch.Tensor):
            return torch.sqrt(x**2 + y**2)
        return np.sqrt(x**2 + y**2)

    return DerivedFieldDef(
        name=out_name,
        description=description,
        typical_range=(0, 1),
        unit=unit,
        is_cyclic=False,
        dependencies=(x_name, y_name),
        compute=compute,
    )


def make_angle_field(x_name: str, y_name: str, out_name: str, description: str, signed: bool = False):
    """Create an angle derived field from two components."""
    def compute(ctx):
        x, y = ctx[x_name], ctx[y_name]
        if isinstance(x, torch.Tensor):
            angle = torch.rad2deg(torch.atan2(y, x))
            return angle if signed else angle % 360
        angle = np.degrees(np.arctan2(y, x))
        return angle if signed else angle % 360

    return DerivedFieldDef(
        name=out_name,
        description=description,
        typical_range=(-180, 180) if signed else (0, 360),
        unit='degrees',
        is_cyclic=True,
        dependencies=(x_name, y_name),
        compute=compute,
    )


# === Example: Electrostatics Field Provider ===
# This would live in the electrostatics solver module, not here

class ElectrostaticsFieldProvider:
    """Field provider for electrostatics PDE."""

    def primary_field_info(self) -> dict[str, FieldInfo]:
        return {
            'ex': FieldInfo('ex', 'Electric field X component', (-1, 1), 'V/m'),
            'ey': FieldInfo('ey', 'Electric field Y component', (-1, 1), 'V/m'),
            'phi': FieldInfo('phi', 'Electric potential', (0, 1), 'V'),
        }

    def derived_field_defs(self) -> dict[str, DerivedFieldDef]:
        return {
            'e_mag': make_magnitude_field('ex', 'ey', 'e_mag', 'Electric field magnitude |E|', 'V/m'),
            'e_angle': make_angle_field('ex', 'ey', 'e_angle', 'Field direction (0-360°)'),
            'e_angle_signed': make_angle_field('ex', 'ey', 'e_angle_signed', 'Field direction (±180°)', signed=True),
        }


# === Example: Fluid Dynamics Field Provider ===

class FluidFieldProvider:
    """Field provider for incompressible Navier-Stokes."""

    def primary_field_info(self) -> dict[str, FieldInfo]:
        return {
            'vx': FieldInfo('vx', 'Velocity X component', (-1, 1), 'm/s'),
            'vy': FieldInfo('vy', 'Velocity Y component', (-1, 1), 'm/s'),
            'p': FieldInfo('p', 'Pressure', (0, 1), 'Pa'),
        }

    def derived_field_defs(self) -> dict[str, DerivedFieldDef]:
        defs = {
            'speed': make_magnitude_field('vx', 'vy', 'speed', 'Flow speed |v|', 'm/s'),
            'v_angle': make_angle_field('vx', 'vy', 'v_angle', 'Flow direction (0-360°)'),
        }
        # Vorticity requires spatial derivatives - more complex
        # defs['vorticity'] = ...
        return defs


# === The LIC Field ===
# Always available regardless of PDE type

LIC_FIELD_INFO = FieldInfo('lic', 'LIC intensity (texture)', (0, 1))


# === Field Context ===

class FieldContext:
    """Manages field access with lazy computation of derived fields.

    This class is PDE-agnostic. It receives field definitions from the solver
    and provides unified access for expression evaluation.

    Usage:
        # Solver provides field metadata
        provider = ElectrostaticsFieldProvider()

        # Create context with primary fields + LIC
        ctx = FieldContext(
            primary_fields={'ex': ex_arr, 'ey': ey_arr, 'phi': phi_arr, 'lic': lic_arr},
            field_info={**provider.primary_field_info(), 'lic': LIC_FIELD_INFO},
            derived_defs=provider.derived_field_defs(),
        )

        # Access fields (derived computed lazily)
        mag = ctx['e_mag']
        angle = ctx['e_angle']
    """

    def __init__(
        self,
        primary_fields: dict[str, np.ndarray | torch.Tensor],
        field_info: dict[str, FieldInfo],
        derived_defs: dict[str, DerivedFieldDef],
    ):
        self._primary = primary_fields
        self._field_info = field_info
        self._derived_defs = derived_defs
        self._cache: dict[str, np.ndarray | torch.Tensor] = {}

        # Detect backend from first field
        first = next(iter(primary_fields.values()))
        self._backend = 'torch' if isinstance(first, torch.Tensor) else 'numpy'

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of fields (H, W)."""
        return next(iter(self._primary.values())).shape

    def field_info(self, name: str) -> FieldInfo | None:
        """Get metadata for a field."""
        if name in self._field_info:
            return self._field_info[name]
        if name in self._derived_defs:
            d = self._derived_defs[name]
            return FieldInfo(d.name, d.description, d.typical_range, d.unit, d.is_cyclic)
        return None

    def __getitem__(self, name: str) -> np.ndarray | torch.Tensor:
        """Get field by name, computing derived fields as needed."""
        # Check primary
        if name in self._primary:
            return self._primary[name]

        # Check cache
        if name in self._cache:
            return self._cache[name]

        # Check for derived field
        if name not in self._derived_defs:
            available = self.available()
            raise KeyError(f"Unknown field: '{name}'. Available: {', '.join(available)}")

        defn = self._derived_defs[name]

        # Check dependencies are available
        for dep in defn.dependencies:
            if dep not in self._primary and dep not in self._cache:
                if dep in self._derived_defs:
                    # Recursively compute dependency
                    _ = self[dep]
                else:
                    raise KeyError(f"Field '{name}' requires '{dep}' which is not available")

        # Compute and cache
        result = defn.compute(self)
        self._cache[name] = result
        return result

    def __contains__(self, name: str) -> bool:
        """Check if field is available (primary or computable)."""
        if name in self._primary:
            return True
        if name in self._derived_defs:
            defn = self._derived_defs[name]
            return all(dep in self for dep in defn.dependencies)
        return False

    def available(self) -> list[str]:
        """List all available field names."""
        result = list(self._primary.keys())
        for name, defn in self._derived_defs.items():
            if name not in result:
                if all(dep in self for dep in defn.dependencies):
                    result.append(name)
        return sorted(result)

    def all_field_info(self) -> dict[str, FieldInfo]:
        """Get metadata for all available fields (for UI)."""
        result = {}
        for name in self.available():
            info = self.field_info(name)
            if info:
                result[name] = info
        return result

    def as_dict(self, names: list[str] | None = None) -> dict:
        """Get fields as dict for expression evaluation.

        If names is None, returns all available fields.
        """
        if names is None:
            names = self.available()
        return {name: self[name] for name in names}


# === Convenience: Create context from solver ===

def create_field_context(
    primary_fields: dict[str, np.ndarray | torch.Tensor],
    lic_field: np.ndarray | torch.Tensor,
    provider: FieldProvider,
) -> FieldContext:
    """Create a FieldContext from a solver's field provider.

    Args:
        primary_fields: Primary fields from solver (ex, ey, phi, etc.)
        lic_field: LIC intensity field (always included)
        provider: FieldProvider from the solver

    Returns:
        Configured FieldContext ready for expression evaluation
    """
    # Merge primary fields with LIC
    all_primary = {**primary_fields, 'lic': lic_field}

    # Merge field info with LIC
    all_info = {**provider.primary_field_info(), 'lic': LIC_FIELD_INFO}

    return FieldContext(
        primary_fields=all_primary,
        field_info=all_info,
        derived_defs=provider.derived_field_defs(),
    )
```

#### channel.py

```python
"""Single-channel configuration for L, C, or H."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import torch

from flowcol.expr import compile as compile_expr, CompiledExpr
from .fields import FieldContext, FIELD_REGISTRY


@dataclass
class ChannelConfig:
    """Configuration for one OKLCH channel.

    An expression is evaluated against field values, normalized to
    an input range, then mapped to an output range.

    Example for L channel:
        expression = "lic"
        input_range = None  # auto: use min/max of result
        output_range = (0.1, 0.9)  # map to L in [0.1, 0.9]

    Example for H channel:
        expression = "e_angle"
        input_range = (0, 360)  # known range of angle
        output_range = (0, 360)  # map directly
        cyclic = True
    """

    expression: str
    input_range: tuple[float, float] | None = None  # None = auto (min/max)
    output_range: tuple[float, float] = (0.0, 1.0)
    cyclic: bool = False  # For H channel: wrap at 360

    # Cached compiled expression (not serialized)
    _compiled: CompiledExpr | None = field(default=None, repr=False, compare=False)

    def compile(self) -> CompiledExpr:
        """Parse and compile the expression."""
        if self._compiled is None:
            self._compiled = compile_expr(self.expression)
        return self._compiled

    def variables(self) -> set[str]:
        """Get variable names used in expression."""
        return self.compile().variables

    def validate(self, available: list[str]) -> list[str]:
        """Validate expression. Returns list of errors."""
        errors = []
        try:
            compiled = self.compile()
            missing = compiled.validate_variables(set(available))
            if missing:
                errors.append(f"Unknown field(s): {', '.join(missing)}")
        except Exception as e:
            errors.append(str(e))
        return errors

    def evaluate(self, ctx: FieldContext) -> np.ndarray | torch.Tensor:
        """Evaluate expression and map to output range.

        Returns array with values in output_range.
        """
        compiled = self.compile()

        # Get only the variables we need
        needed = compiled.variables
        field_dict = ctx.as_dict(list(needed))

        # Evaluate
        if ctx.backend == 'torch':
            raw = compiled.eval_torch(field_dict)
        else:
            raw = compiled.eval_numpy(field_dict)

        # Normalize to [0, 1] based on input_range
        if self.input_range is None:
            # Auto: use actual min/max
            if ctx.backend == 'torch':
                vmin, vmax = raw.min().item(), raw.max().item()
            else:
                vmin, vmax = float(raw.min()), float(raw.max())
        else:
            vmin, vmax = self.input_range

        if vmax > vmin:
            normalized = (raw - vmin) / (vmax - vmin)
        else:
            normalized = raw * 0  # All same value → 0

        # Clamp to [0, 1] (unless cyclic)
        if not self.cyclic:
            if ctx.backend == 'torch':
                normalized = torch.clamp(normalized, 0, 1)
            else:
                normalized = np.clip(normalized, 0, 1)

        # Map to output range
        out_min, out_max = self.output_range
        result = normalized * (out_max - out_min) + out_min

        # Handle cyclic (for hue)
        if self.cyclic and out_max > out_min:
            period = out_max - out_min
            if ctx.backend == 'torch':
                result = torch.remainder(result - out_min, period) + out_min
            else:
                result = np.mod(result - out_min, period) + out_min

        return result


# === Convenience Constructors ===

def L(expr: str, out_range: tuple[float, float] = (0.0, 1.0),
      in_range: tuple[float, float] | None = None) -> ChannelConfig:
    """Create Lightness channel config."""
    return ChannelConfig(expr, input_range=in_range, output_range=out_range, cyclic=False)

def C(expr: str, out_range: tuple[float, float] = (0.0, 0.35),
      in_range: tuple[float, float] | None = None) -> ChannelConfig:
    """Create Chroma channel config.

    Default output range (0, 0.35) is conservative to avoid gamut issues.
    """
    return ChannelConfig(expr, input_range=in_range, output_range=out_range, cyclic=False)

def H(expr: str, out_range: tuple[float, float] = (0.0, 360.0),
      in_range: tuple[float, float] | None = None) -> ChannelConfig:
    """Create Hue channel config."""
    return ChannelConfig(expr, input_range=in_range, output_range=out_range, cyclic=True)
```

#### compose.py

```python
"""OKLCH composition: fields → RGB."""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch

from flowcol.colorspace import oklch_to_srgb, gamut_compress
from .channel import ChannelConfig, L, C, H
from .fields import FieldContext


@dataclass
class OKLCHConfig:
    """Full OKLCH color configuration."""

    L: ChannelConfig
    C: ChannelConfig
    H: ChannelConfig
    gamut_method: Literal['compress', 'clip'] = 'compress'

    def validate(self, available: list[str]) -> list[str]:
        """Validate all channels. Returns list of errors."""
        errors = []
        for name, channel in [('L', self.L), ('C', self.C), ('H', self.H)]:
            for err in channel.validate(available):
                errors.append(f"{name}: {err}")
        return errors

    def variables(self) -> set[str]:
        """Get all variable names used across all channels."""
        return self.L.variables() | self.C.variables() | self.H.variables()


def compose_oklch(
    fields: FieldContext,
    config: OKLCHConfig,
) -> np.ndarray | torch.Tensor:
    """Compose RGB image from fields using OKLCH color mapping.

    Args:
        fields: Field context with primary fields loaded
        config: OKLCH channel configuration

    Returns:
        RGB array/tensor (H, W, 3) with values in [0, 1]
    """
    # Evaluate each channel
    L_vals = config.L.evaluate(fields)
    C_vals = config.C.evaluate(fields)
    H_vals = config.H.evaluate(fields)

    # Gamut mapping
    L_vals, C_vals, H_vals = gamut_compress(
        L_vals, C_vals, H_vals,
        method=config.gamut_method
    )

    # Convert to sRGB
    rgb = oklch_to_srgb(L_vals, C_vals, H_vals)

    return rgb


def compose_oklch_uint8(
    fields: FieldContext,
    config: OKLCHConfig,
) -> np.ndarray:
    """Compose RGB uint8 image from fields.

    Convenience wrapper that returns uint8 numpy array for display.
    """
    rgb_float = compose_oklch(fields, config)

    if isinstance(rgb_float, torch.Tensor):
        rgb_float = rgb_float.cpu().numpy()

    return (rgb_float * 255).clip(0, 255).astype(np.uint8)
```

#### presets.py — PDE-Specific Preset Collections

Presets are organized by PDE type since they reference PDE-specific field names.

```python
"""Built-in OKLCH presets for common artistic effects.

Presets are PDE-specific because they reference field names like 'e_angle'
(electrostatics) or 'v_angle' (fluids). Each PDE type has its own preset
collection.

Universal presets (using only 'lic') work with any PDE.
"""

from .channel import L, C, H
from .compose import OKLCHConfig


# === Universal Presets (work with any PDE) ===

UNIVERSAL_PRESETS: dict[str, OKLCHConfig] = {
    'Grayscale': OKLCHConfig(
        L=L('lic'),
        C=C('0'),
        H=H('0'),
    ),
    'High Contrast B&W': OKLCHConfig(
        L=L('pow(lic, 0.6)', out_range=(0.0, 1.0)),
        C=C('0'),
        H=H('0'),
    ),
}


# === Electrostatics Presets ===
# Use fields: lic, ex, ey, phi, e_mag, e_angle

ELECTROSTATICS_PRESETS: dict[str, OKLCHConfig] = {
    'Direction Rainbow': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('0.15'),
        H=H('e_angle', in_range=(0, 360)),
    ),
    'Subtle Direction': OKLCHConfig(
        L=L('lic', out_range=(0.1, 0.9)),
        C=C('0.06'),
        H=H('e_angle', in_range=(0, 360)),
    ),
    'Magnitude Luminance': OKLCHConfig(
        L=L('e_mag', out_range=(0.15, 0.85)),
        C=C('0.12'),
        H=H('e_angle', in_range=(0, 360)),
    ),
    'Potential Rainbow': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('0.18'),
        H=H('phi', in_range=(0, 1), out_range=(0, 300)),
    ),
    'Field Intensity': OKLCHConfig(
        L=L('lic * 0.7 + e_mag * 0.3', out_range=(0.1, 0.9)),
        C=C('e_mag * 0.15', out_range=(0, 0.2)),
        H=H('e_angle', in_range=(0, 360)),
    ),
    # ... more electrostatics presets
}


# === Fluid Dynamics Presets ===
# Use fields: lic, vx, vy, p, speed, v_angle

FLUID_PRESETS: dict[str, OKLCHConfig] = {
    'Flow Direction': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('0.15'),
        H=H('v_angle', in_range=(0, 360)),
    ),
    'Speed Luminance': OKLCHConfig(
        L=L('speed', out_range=(0.15, 0.85)),
        C=C('0.12'),
        H=H('v_angle', in_range=(0, 360)),
    ),
    'Pressure Hue': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('0.18'),
        H=H('p', in_range=(0, 1), out_range=(240, 0)),  # blue=high, red=low
    ),
    # ... more fluid presets
}


# === Heat Equation Presets ===
# Use fields: lic, T, qx, qy, q_mag, q_angle

HEAT_PRESETS: dict[str, OKLCHConfig] = {
    'Temperature': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('0.2'),
        H=H('T', in_range=(0, 1), out_range=(240, 30)),  # blue=cold, orange=hot
    ),
    'Heat Flow': OKLCHConfig(
        L=L('lic', out_range=(0.3, 0.7)),
        C=C('q_mag * 0.2', out_range=(0, 0.25)),
        H=H('q_angle', in_range=(0, 360)),
    ),
    # ... more heat presets
}


# === Preset Registry by PDE Type ===

PRESETS_BY_PDE: dict[str, dict[str, OKLCHConfig]] = {
    'universal': UNIVERSAL_PRESETS,
    'electrostatics': ELECTROSTATICS_PRESETS,
    'fluid': FLUID_PRESETS,
    'heat': HEAT_PRESETS,
}


def get_presets_for_pde(pde_type: str) -> dict[str, OKLCHConfig]:
    """Get all presets applicable to a PDE type.

    Returns universal presets merged with PDE-specific presets.
    """
    result = dict(UNIVERSAL_PRESETS)
    if pde_type in PRESETS_BY_PDE:
        result.update(PRESETS_BY_PDE[pde_type])
    return result


def list_presets(pde_type: str) -> list[str]:
    """List preset names for a PDE type."""
    return list(get_presets_for_pde(pde_type).keys())


def get_preset(name: str, pde_type: str) -> OKLCHConfig:
    """Get preset by name for a PDE type."""
    presets = get_presets_for_pde(pde_type)
    if name not in presets:
        raise KeyError(f"Unknown preset '{name}' for PDE type '{pde_type}'")
    return presets[name]


def validate_preset_for_fields(preset: OKLCHConfig, available_fields: list[str]) -> list[str]:
    """Check if a preset's expressions use only available fields.

    Returns list of missing fields (empty if valid).
    """
    needed = preset.variables()
    return [f for f in needed if f not in available_fields]
```

#### config.py

```python
"""Top-level color configuration."""

from dataclasses import dataclass, field
from typing import Literal

from .compose import OKLCHConfig
from .presets import PRESETS, get_preset


@dataclass
class ColorConfig:
    """Unified color configuration supporting both palette and OKLCH modes.

    For backward compatibility, defaults to palette mode with existing settings.
    """

    mode: Literal['palette', 'oklch'] = 'palette'

    # === Palette Mode (existing) ===
    palette_name: str = 'Ink Wash'
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    clip_percent: float = 0.5

    # === OKLCH Mode ===
    oklch_preset: str | None = None       # Use named preset
    oklch_custom: OKLCHConfig | None = None  # Or custom config

    def get_oklch(self) -> OKLCHConfig:
        """Get effective OKLCH config (preset or custom)."""
        if self.oklch_preset and self.oklch_preset in PRESETS:
            return get_preset(self.oklch_preset)
        if self.oklch_custom:
            return self.oklch_custom
        # Fallback
        return get_preset('Grayscale')

    # === Serialization ===

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        d = {
            'mode': self.mode,
            'palette_name': self.palette_name,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'gamma': self.gamma,
            'clip_percent': self.clip_percent,
        }
        if self.oklch_preset:
            d['oklch_preset'] = self.oklch_preset
        if self.oklch_custom:
            d['oklch_custom'] = _oklch_config_to_dict(self.oklch_custom)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ColorConfig':
        """Deserialize from dict."""
        oklch_custom = None
        if 'oklch_custom' in data:
            oklch_custom = _oklch_config_from_dict(data['oklch_custom'])

        return cls(
            mode=data.get('mode', 'palette'),
            palette_name=data.get('palette_name', 'Ink Wash'),
            brightness=data.get('brightness', 0.0),
            contrast=data.get('contrast', 1.0),
            gamma=data.get('gamma', 1.0),
            clip_percent=data.get('clip_percent', 0.5),
            oklch_preset=data.get('oklch_preset'),
            oklch_custom=oklch_custom,
        )


def _channel_to_dict(ch) -> dict:
    return {
        'expression': ch.expression,
        'input_range': ch.input_range,
        'output_range': ch.output_range,
        'cyclic': ch.cyclic,
    }

def _channel_from_dict(d: dict):
    from .channel import ChannelConfig
    return ChannelConfig(
        expression=d['expression'],
        input_range=tuple(d['input_range']) if d.get('input_range') else None,
        output_range=tuple(d.get('output_range', (0, 1))),
        cyclic=d.get('cyclic', False),
    )

def _oklch_config_to_dict(cfg: OKLCHConfig) -> dict:
    return {
        'L': _channel_to_dict(cfg.L),
        'C': _channel_to_dict(cfg.C),
        'H': _channel_to_dict(cfg.H),
        'gamut_method': cfg.gamut_method,
    }

def _oklch_config_from_dict(d: dict) -> OKLCHConfig:
    return OKLCHConfig(
        L=_channel_from_dict(d['L']),
        C=_channel_from_dict(d['C']),
        H=_channel_from_dict(d['H']),
        gamut_method=d.get('gamut_method', 'compress'),
    )
```

#### pipeline.py

```python
"""Unified color pipeline entry point."""

import numpy as np
import torch

from .config import ColorConfig
from .fields import FieldContext
from .compose import compose_oklch


def build_rgb(
    fields: dict[str, np.ndarray | torch.Tensor],
    config: ColorConfig,
) -> np.ndarray:
    """Build RGB image from fields using configured color mode.

    This is the main entry point for colorization.

    Args:
        fields: Dict of field arrays.
            - Palette mode: only 'lic' required
            - OKLCH mode: should include 'lic', 'ex', 'ey', 'phi'
        config: Color configuration

    Returns:
        RGB uint8 array (H, W, 3)
    """
    if config.mode == 'palette':
        return _build_rgb_palette(fields['lic'], config)
    else:
        return _build_rgb_oklch(fields, config)


def _build_rgb_palette(lic: np.ndarray | torch.Tensor, config: ColorConfig) -> np.ndarray:
    """Palette-mode colorization (existing code path)."""
    # Import existing implementation
    from flowcol.postprocess.color import build_base_rgb, ColorParams

    params = ColorParams(
        clip_percent=config.clip_percent,
        brightness=config.brightness,
        contrast=config.contrast,
        gamma=config.gamma,
        color_enabled=True,  # palette mode always uses color
        palette=config.palette_name,
    )

    if isinstance(lic, torch.Tensor):
        lic = lic.cpu().numpy()

    return build_base_rgb(lic, params)


def _build_rgb_oklch(
    fields: dict[str, np.ndarray | torch.Tensor],
    config: ColorConfig,
) -> np.ndarray:
    """OKLCH-mode colorization."""
    ctx = FieldContext(fields)
    oklch_config = config.get_oklch()

    # Validate
    errors = oklch_config.validate(ctx.available())
    if errors:
        raise ValueError(f"Invalid OKLCH config: {'; '.join(errors)}")

    # Compose
    rgb_float = compose_oklch(ctx, oklch_config)

    # Convert to uint8
    if isinstance(rgb_float, torch.Tensor):
        rgb_float = rgb_float.cpu().numpy()

    return (rgb_float * 255).clip(0, 255).astype(np.uint8)
```

---

## Data Flow

### Current Flow

```
PDE Solver → Ex, Ey, φ → LIC → colorize(lic) → RGB
```

Only LIC intensity reaches colorization.

### New Flow

```
PDE Solver
    │
    ├─→ Ex, Ey, φ ──────────────┐
    │                           │
    └─→ LIC Computation         │
            │                   │
            ▼                   ▼
        ┌─────────────────────────┐
        │   Field Bundle          │
        │   {lic, ex, ey, phi}    │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   FieldContext          │
        │   (lazy derived fields) │
        └─────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    L expr      C expr      H expr
        │           │           │
        └─────┬─────┴─────┬─────┘
              │           │
              ▼           ▼
        ┌───────────────────────┐
        │   Gamut Mapping       │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   OKLCH → sRGB        │
        └───────────────────────┘
                    │
                    ▼
               RGB (H,W,3)
```

### Integration Points

The main change to existing code: pass field bundle instead of just LIC.

```python
# Before (in render orchestrator or similar)
rgb = build_base_rgb(lic_array, color_params)

# After
from flowcol.color import build_rgb, ColorConfig

config = ColorConfig(
    mode='oklch',
    oklch_preset='Direction Rainbow',
)

fields = {
    'lic': lic_array,
    'ex': ex_array,
    'ey': ey_array,
    'phi': phi_array,
}

rgb = build_rgb(fields, config)
```

---

## GPU Strategy

All color operations should run on GPU when fields are torch tensors.

### Automatic Backend Selection

The FieldContext and expression compiler automatically detect whether inputs are numpy or torch:

```python
ctx = FieldContext({'lic': lic_tensor, 'ex': ex_tensor, ...})
# ctx.backend == 'torch'

result = channel.evaluate(ctx)
# Uses torch operations, stays on GPU
```

### GPU-Specific Optimizations

For `gpu/colorspace.py`:

```python
def oklch_to_srgb_gpu(L: Tensor, C: Tensor, H: Tensor) -> Tensor:
    """OKLCH → sRGB on GPU.

    Optimized for batch processing of image data.
    """
    # All operations are element-wise, perfect for GPU
    H_rad = H * (torch.pi / 180)
    a = C * torch.cos(H_rad)
    b = C * torch.sin(H_rad)

    # Matrix operations for OKLab → Linear RGB
    # ... (see colorspace/oklch.py for math)

    return rgb
```

### Memory Considerations

For a 4K image (3840×2160):
- Each field: ~33 MB (float32)
- 4 primary fields: ~132 MB
- 3 derived fields (if all computed): ~99 MB
- RGB output: ~99 MB

Total: ~330 MB GPU memory for full pipeline. This is fine for modern GPUs.

For memory-constrained cases, derived fields are computed lazily and could be evicted after use.

---

## Expression Language Reference

### Variables (Fields)

Available fields depend on the active PDE solver. The UI should display available fields dynamically.

**Universal (always available):**

| Name | Description | Typical Range |
|------|-------------|---------------|
| `lic` | LIC intensity (texture) | [0, 1] |

**Electrostatics example:**

| Name | Description | Typical Range |
|------|-------------|---------------|
| `ex`, `ey` | E-field components | [-1, 1] |
| `phi` | Electric potential | [0, 1] |
| `e_mag` | Field magnitude \|E\| | [0, 1] |
| `e_angle` | Field direction | [0, 360) degrees |

**Fluid dynamics example:**

| Name | Description | Typical Range |
|------|-------------|---------------|
| `vx`, `vy` | Velocity components | [-1, 1] |
| `p` | Pressure | [0, 1] |
| `speed` | Flow speed \|v\| | [0, 1] |
| `v_angle` | Flow direction | [0, 360) degrees |

**Heat equation example:**

| Name | Description | Typical Range |
|------|-------------|---------------|
| `T` | Temperature | [0, 1] |
| `qx`, `qy` | Heat flux components | [-1, 1] |
| `q_mag` | Heat flux magnitude | [0, 1] |
| `q_angle` | Heat flow direction | [0, 360) degrees |

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `lic + 0.1` |
| `-` | Subtraction | `1 - lic` |
| `*` | Multiplication | `lic * 2` |
| `/` | Division | `e_mag / 2` |
| `^` | Power | `lic ^ 0.5` |
| `-` (unary) | Negation | `-e_angle` |

### Functions

| Function | Args | Description |
|----------|------|-------------|
| `abs(x)` | 1 | Absolute value |
| `sqrt(x)` | 1 | Square root |
| `pow(x, y)` | 2 | Power (x^y) |
| `sin(x)`, `cos(x)` | 1 | Trig (radians) |
| `atan2(y, x)` | 2 | Two-arg arctangent |
| `deg(x)` | 1 | Radians → degrees |
| `rad(x)` | 1 | Degrees → radians |
| `mod(x, y)` | 2 | Modulo |
| `min(x, y)`, `max(x, y)` | 2 | Element-wise min/max |
| `clamp(x, lo, hi)` | 3 | Clamp to range |
| `lerp(a, b, t)` | 3 | Linear interpolation |
| `smoothstep(lo, hi, x)` | 3 | Smooth interpolation |
| `sign(x)` | 1 | Sign (-1, 0, 1) |
| `floor(x)`, `ceil(x)` | 1 | Rounding |
| `frac(x)` | 1 | Fractional part |

### Example Expressions

```python
# Simple field reference
"lic"
"e_angle"

# Gamma correction
"pow(lic, 0.8)"

# Weighted combination
"lic * 0.7 + e_mag * 0.3"

# Angle manipulation
"mod(e_angle + 90, 360)"

# Conditional-like (using min/max)
"max(lic, e_mag)"

# Smooth thresholding
"smoothstep(0.3, 0.7, lic)"

# Field magnitude from components (same as e_mag)
"sqrt(ex^2 + ey^2)"
```

---

## UI Considerations

### Mode Selection

```
Color Mode: [● Palette ○ OKLCH]
```

When OKLCH selected, show additional controls.

### Preset Selection

```
Preset: [▼ Direction Rainbow     ]

  Preview: [color swatch / mini preview]
```

### Advanced Mode

Show when user clicks "Customize" or similar:

```
┌─ Lightness ────────────────────────────┐
│ Expression: [lic                     ] │
│ Input range: [auto ▼]                  │
│ Output: [0.10] to [0.90]               │
└────────────────────────────────────────┘

┌─ Chroma ───────────────────────────────┐
│ Expression: [e_mag * 0.2             ] │
│ Input range: [auto ▼]                  │
│ Output: [0.00] to [0.25]               │
└────────────────────────────────────────┘

┌─ Hue ──────────────────────────────────┐
│ Expression: [e_angle                 ] │
│ Input range: [0   ] to [360 ]          │
│ Output: [0   ] to [360 ]  ☑ Cyclic     │
└────────────────────────────────────────┘

[Gamut: Compress ▼]  [Save as Preset...]
```

### Expression Help

Pop-up or sidebar showing available fields and functions with examples.

### Error Display

Inline validation errors:
```
Expression: [e_maf                    ]
            ⚠ Unknown field 'e_maf'. Did you mean 'e_mag'?
```

---

## Testing Strategy

### Unit Tests

**colorspace/**
- Round-trip accuracy: sRGB → OKLCH → sRGB within tolerance
- Compare against reference implementation (Björn Ottosson's)
- Edge cases: black, white, primary colors, out-of-gamut

**expr/**
- Parser: valid expressions, syntax errors, edge cases
- Compiler: evaluation correctness, numpy/torch parity
- Functions: each function correct, numpy/torch match

**color/**
- FieldContext: lazy evaluation, caching, missing field errors
- ChannelConfig: normalization, cyclic handling
- Presets: all presets produce valid RGB

### Integration Tests

- Full pipeline: fields → RGB for various configs
- GPU/CPU parity: same config produces same output (within float tolerance)
- Serialization round-trip: config → dict → config → same RGB

### Visual Tests

- Reference images for each preset
- Gamut mapping visual verification (no harsh clipping artifacts)

---

## Migration Plan

### Phase 1: Foundation (No Breaking Changes)

Add new modules without touching existing code:

```
flowcol/
├── colorspace/     # NEW
├── expr/           # NEW
├── color/          # NEW
└── ... existing unchanged ...
```

Validate independently with unit tests.

### Phase 2: Integration

Wire up OKLCH path alongside existing palette path:

1. Add `ColorConfig` to project settings
2. Update render pipeline to pass field bundle
3. Add UI toggle for OKLCH mode
4. Preset selector in UI

Existing palette mode continues to work unchanged.

### Phase 3: Polish

1. Advanced expression editor UI
2. Custom preset saving
3. Performance optimization (gamut LUT, etc.)
4. Documentation and examples

### Phase 4: Optional Cleanup

Move palette code from `render.py` to `color/palette.py`:
- Keeps all color logic in one place
- `render.py` becomes purely about LIC computation

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Gamut mapping artifacts | Medium | Conservative defaults, multiple methods |
| Expression parser edge cases | Low | Extensive test suite, clear error messages |
| UI complexity | Medium | Good presets reduce need for manual config |
| Performance regression | Low | GPU path, lazy field computation |
| Breaking existing projects | Low | Palette mode unchanged, migration optional |

---

## Open Questions

1. **Gamut LUT**: Should we precompute max-chroma lookup table? Adds ~360KB but speeds up gamut compression significantly.

2. **PDE-specific derived fields**: Each PDE type may want specialized derived fields:
   - Electrostatics: divergence ∇·E, distance to conductors
   - Fluids: vorticity ∇×v, strain rate, Q-criterion
   - Heat: Laplacian ∇²T
   - General: spatial derivatives require grid spacing info

3. **Expression DSL extensions**: Would any of these be valuable?
   - Conditional: `if(cond, then, else)`
   - Noise: `noise(x, y, scale)`
   - Spatial filters: `blur(field, sigma)`

4. **Preset discovery UX**: How do users find presets when switching PDEs?
   - Show only compatible presets?
   - Gray out incompatible ones with explanation?
   - Auto-switch to universal preset when changing PDE?

5. **Per-region color**: Should OKLCH mode support per-region overrides like palette mode does?

6. **Custom user presets**: Should users be able to save presets?
   - Per-project vs. global?
   - How to handle presets that reference PDE-specific fields?

7. **Field provider registration**: How do new PDE solvers register their fields?
   - Protocol/interface approach (current design)
   - Plugin/discovery system?
   - Configuration file?

---

## References

- [Oklab color space](https://bottosson.github.io/posts/oklab/) - Björn Ottosson's original post
- [OKLCH in CSS](https://developer.chrome.com/articles/oklch-in-css-why-quit-using-hsl/) - Practical OKLCH overview
- [Gamut mapping](https://bottosson.github.io/posts/gamutclipping/) - Björn Ottosson's gamut clipping
- [Color.js](https://colorjs.io/) - Reference implementation

---

## Appendix: OKLCH Color Space

### What is OKLCH?

OKLCH is a cylindrical representation of the Oklab color space:

- **L** (Lightness): 0 = black, 1 = white
- **C** (Chroma): 0 = gray, higher = more saturated
- **H** (Hue): 0-360 degrees around the color wheel

### Why Perceptual Uniformity Matters

In RGB or HSL, "equal steps" don't look equal:

```
HSL hue 0° → 60° → 120° (red → yellow → green)
```

Yellow appears much brighter than red or green at the same L value.

In OKLCH, equal numerical changes produce equal perceived changes:

```
OKLCH L=0.5, C=0.15, H varies
```

All hues appear equally bright because L is constant.

### Gamut Limitations

sRGB can't display all OKLCH colors. The valid gamut is an irregular 3D shape:

- At L=0 (black) and L=1 (white): only C=0 is valid
- At L=0.5: maximum C varies by hue (~0.32 for some hues)
- Highly saturated colors at extreme lightness are out-of-gamut

This is why gamut mapping is essential.
