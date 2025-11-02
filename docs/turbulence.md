# Turbulence (Curl‑Noise) Design

Status: Proposal (Edit‑time controls, pre‑render)

## Goal

Add an optional, divergence‑free turbulence layer as a curl‑noise vector field that perturbs the solved electric field before LIC. Keep controls expressive but low‑dimensional, support targeting of hollow interiors and the ambient exterior (complement of conductor + interior), and preserve the project’s physics/UX model.

## Summary

- Compute φ and E as today. Immediately before LIC, add a masked curl‑noise field `n = (nx, ny)` to E.
- Controls live in the Edit panel (Conductor Controls), not post‑processing.
- Global controls set the default behavior; per‑conductor overrides apply to hollow interiors.
- Targeting supports: Interior (selected or per‑conductor enabled), Exterior (complement of all conductors), and Everywhere (ambient).
- Curl‑noise is deterministic (seeded), divergence‑free, and normalized to a predictable amplitude. Optional alignment to the solved field and a global drift direction are supported as advanced toggles.

---

## User Experience

Edit panel (where users set voltages and select conductors):

- Turbulence (global)
  - Enabled [ ]
  - Strength (wet/dry)
  - Scale (dominant wavelength)
  - Seed
  - Target: Everywhere / Exterior / Interior (selected)
  - Edge Falloff (smooth feather distance)
  - Advanced (collapsed by default):
    - Alignment: Off / Perpendicular / Parallel
    - Rotation Bias (−1…+1)
    - Drift Angle (0–360°)

- Per‑conductor (visible when a conductor is selected)
  - Interior Turbulence [ ]
  - Strength (override)

Mental model:

- “Turn turbulence on, pick how strong/coarse it is, choose where it lives (cavity vs outside), optionally bias the swirl/flow relationship.”

---

## Data Model

In `flowcol/types.py` add global project turbulence settings and per‑conductor interior overrides.

```python
@dataclass
class TurbulenceSettings:
    enabled: bool = False
    strength: float = 0.3              # wet/dry mix
    scale_px: float = 24.0             # dominant wavelength (compute grid pixels)
    seed: int = 0
    edge_falloff_px: float = 6.0       # feather at region borders
    target: str = "exterior"           # "everywhere" | "exterior" | "interior"
    alignment_mode: str = "off"        # "off" | "perp" | "parallel"
    rotation_bias: float = 0.0         # −1…+1 (swirl bias)
    drift_angle_deg: float = 0.0       # 0–360

@dataclass
class Conductor:
    # existing fields…
    interior_turbulence_enabled: bool = False
    interior_turbulence_strength: float | None = None  # None => use global strength

@dataclass
class Project:
    # existing fields…
    turbulence: TurbulenceSettings = field(default_factory=TurbulenceSettings)
```

Serialization (`flowcol/serialization.py`):

- Include `project.turbulence` in `_project_to_dict/_dict_to_project` with defaults if missing (backward‑compatible).
- Extend conductor entries with `interior_turbulence_enabled` and `interior_turbulence_strength` (optional).

Actions (`flowcol/app/actions.py`):

- Add mutators: `set_turbulence_enabled/strength/scale/seed/edge_falloff/target/alignment/rotation_bias/drift_angle` and `set_conductor_interior_turbulence_enabled/strength`. All mark `field_dirty = True`, `render_dirty = True`.

---

## UI Wiring (Edit Panel)

`flowcol/ui/dpg/conductor_controls_panel.py`:

- In `build_conductor_controls_container`, add a “Field Turbulence” group below voltages/smooth:
  - Global controls as above, mapped to `actions.set_turbulence_*`.
  - Per‑conductor interior toggle + strength override, mapped to `actions.set_conductor_interior_turbulence_*` (only visible when a conductor is selected).

UX specifics:

- “Interior (selected)” target uses the currently selected conductor’s `interior_mask`. If none selected, show a hint (“Select a conductor with an interior”).
- “Exterior” target uses the complement of all conductors’ surface∪interior masks.
- “Everywhere (ambient)” means the whole domain except surface masks (conductor bodies) unless users explicitly want conductor body turbulence (future option).

---

## Pipeline Integration

Hook in `flowcol/pipeline.py:perform_render` immediately after computing E and before LIC.

1) Compute field as usual:

```python
ex, ey = compute_field(...)
```

2) If `project.turbulence.enabled`:

- Rasterize masks at render resolution (aligned with crop) via `rasterize_conductor_masks` which already returns `surface_masks_canvas` and `interior_masks_canvas` in `RenderResult`.
- Build a float region weight `W` in [0,1]:
  - Interior: union of selected interior mask; if none selected but per‑conductor interior turbulence is enabled on some conductors, union those interiors.
  - Exterior: complement of union(surface ∪ interior) across all conductors.
  - Everywhere: ones minus surface masks (by default; keep this choice explicit in UI copy).
- Feather edges: use a distance transform and smoothstep to avoid seams:

```python
# d = distance (pixels) to the nearest border where turbulence should fade
w = smoothstep(0, edge_falloff_px, d)  # clamp to [0,1]
W *= w
```

- Generate curl‑noise `(nx, ny)` at full compute resolution using `noise.generate_curl_noise(...)` (see below).
- Optional alignment: project onto perpendicular/parallel to E:

```python
if mode == "perp":   n := n - dot(n, ê)*ê
if mode == "parallel": n := dot(n, ê)*ê
```

- Strength mixing (absolute in MVP):

```python
ex += strength * W * nx
ey += strength * W * ny
```

3) Proceed with LIC unchanged.

Preview mode note: turbulence is generated at the final compute resolution (after any Poisson preview upsampling/relaxation) to keep spatial frequencies stable.

---

## Curl‑Noise Generation

New module: `flowcol/noise.py`.

Functions:

```python
def generate_curl_noise(shape, scale_px, seed, *, octaves=1, rotation_bias=0.0, drift_angle_deg=0.0) -> tuple[np.ndarray, np.ndarray]
```

Algorithm:

1) Scalar field ψ: start with white noise, low‑pass via Gaussian with `sigma ≈ scale_px / c` (e.g., c≈2–3), or combine 2–3 octaves: decreasing sigma and amplitudes.
2) Compute gradients via central differences: `(gx, gy) = ∇ψ`.
3) Apply drift: rotate `(gx, gy)` by `θ = drift_angle_deg`.
4) Form curl vector (2D stream function curl): `nx = +gy`, `ny = −gx` (divergence‑free by construction).
5) Apply rotation bias b ∈ [−1, +1]: `n_biased = normalize((1−|b|)*n + b*R90(n))` where `R90` rotates by +90°; sign of b controls swirl sense.
6) Normalize per‑pixel with ε guard to keep amplitude predictable; output float32.

Alignment helper:

```python
def project_align(nx, ny, ex, ey, mode) -> tuple[np.ndarray, np.ndarray]
```

Implements perpendicular/parallel projection with |E| thresholding to avoid instability at field nulls.

Feathering helper:

```python
def apply_falloff(nx, ny, weight_mask, edge_falloff_px)
```

Uses `distance_transform_edt` and a smoothstep to multiply the vectors by a soft border mask.

Determinism:

- Seed all RNG paths using `(seed, shape)`.
- Cache ψ in a small LRU keyed by `(shape, seed, scale_px, octaves)` to keep scrubbing smooth.

---

## Targeting & Masks

- Use `rasterize_conductor_masks` (already used for GPU postprocess) to obtain surface and interior masks aligned with the cropped render (`offset_x/offset_y`).
- Interior target: union of selected interior(s). If no selection and any per‑conductor interior turbulence is enabled, union those.
- Exterior target: complement of union(surface ∪ interior) across all conductors.
- Everywhere: ones minus surface masks (default behavior keeps conductor bodies calm).
- Always feather with `edge_falloff_px` to avoid visible seams.

---

## Performance & Caching

- Cost: a Gaussian blur plus 2 gradient passes over the compute grid — negligible vs Poisson + LIC.
- Generate turbulence once per render (at compute resolution), independent of Poisson preview scale.
- Optional in‑memory LRU cache for ψ to reduce recomputation during slider scrubbing.

---

## Subtleties & Pitfalls

- Units: define `scale_px` in compute‑grid pixels (not canvas pixels) so appearance is consistent with resolution multipliers.
- Alignment at |E|≈0: clamp normalization with ε and reduce turbulence contribution near field nulls if needed.
- Mask alignment: always rely on `rasterize_conductor_masks` with the same `margin/scale/offset` used for the crop.
- Feathering: avoid binary masks; use float masks + smoothstep over distance to eliminate seams.
- Multiple conductors: for overlapping interior selections blend by `max` of strengths rather than sum to avoid runaway amplitudes.
- Determinism: include `seed` in cache keys; ensure identical outputs for same `(shape, seed, scale)`.
- Boundaries: Neumann edges as emitters are a potential future option; off by default to match expectations.

---

## MVP Scope (Phase 1)

- Global controls: Enabled, Strength, Scale, Seed, Target, Edge Falloff.
- Per‑conductor overrides: Interior Turbulence [ ], Strength.
- No alignment/drift/bias in MVP (add in Phase 2 under Advanced).

## Phase 2

- Add Alignment, Rotation Bias, and Drift Angle.
- Optional: turbulence weighting by local |E| and/or conductor voltage magnitude (single toggle), Neumann edge sources.
- GPU implementation of curl‑noise if needed for very large fields.

---

## Work Items (Files)

- Add `flowcol/noise.py` (curl‑noise + helpers).
- Update `flowcol/types.py` (TurbulenceSettings; Project.turbulence; Conductor interior overrides).
- Update `flowcol/serialization.py` (save/load project turbulence and per‑conductor overrides).
- Update `flowcol/app/actions.py` (mutators for turbulence; mark field/render dirty).
- Update `flowcol/ui/dpg/conductor_controls_panel.py` (edit‑time turbulence controls; selected‑conductor overrides).
- Update `flowcol/pipeline.py` (inject turbulence before LIC; use masks and feathering; keep Poisson unchanged).
- Optionally update `flowcol/defaults.py` (turbulence defaults).

---

## Testing & Acceptance

Unit tests (CPU):

- Divergence check: finite‑difference estimate of ∇·n is near 0 within tolerance.
- Determinism: same `(shape, seed, scale)` → identical `(nx, ny)`.
- Targeting: turbulence magnitude ~0 outside selected region; non‑zero inside; exterior leaves conductor bodies calm.
- Feathering: monotonic decay near borders for chosen `edge_falloff`.
- Alignment (Phase 2): parallel/perpendicular projections behave as expected (dot product checks).

Integration checks:

- Small scenes (disk, ring):
  - Interior target shows turbulence inside cavity only.
  - Exterior target swirls outside shell; shell remains calm.
  - “Everywhere” applies to ambient (excluding shell body by default).
  - Toggling Enabled is deterministic/stable; seed changes pattern only.
  - Render times increase minimally vs baseline.

Acceptance criteria:

- Controls live in Edit panel and re‑render on change.
- Visuals show expected masked turbulence with smooth edges.
- Old projects load intact; turbulence defaults to off.
- No change to Poisson correctness; LIC inputs remain finite and stable.

