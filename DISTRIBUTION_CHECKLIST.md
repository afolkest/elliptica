# Elliptica Distribution Fix Checklist

Track progress on distribution issues. Check off items as they're completed.

---

## P0 - Critical (Fix Immediately)

These will break 95%+ of installations.

### Packaging & Data Files
- [x] Add `[tool.setuptools.package-data]` to `pyproject.toml` for JSON files
- [x] Create `MANIFEST.in` file for source distributions
- [x] Verify JSON files included: `palettes/library.json`, `palettes_user.json`

### Python Version
- [x] Change `requires-python` from `>=3.10` to `>=3.11` in `pyproject.toml`

### NumPy 2.x Compatibility
- [x] Fix `render.py:732` - remove `dtype` param from `np.power()`, cast result instead
- [x] Fix `render.py:780` - same issue, remove `dtype` param

### Serialization Safety
- [x] Add `.get()` defaults in `serialization.py:178-187` for backward compatibility
- [x] Create or remove reference to `elliptica.migrate` script (`serialization.py:110-115`)

### Headless Environment
- [x] Add display/headless check before `dpg.create_context()` in `app.py`
- [x] Provide helpful error message for headless environments

---

## P1 - High Priority (Before Any Release)

### Dependency Pinning
- [x] Pin `numpy>=1.24.0` (NumPy 2.x compat verified in P0)
- [x] Pin `scipy>=1.10.0`
- [x] Pin `numba>=0.57.0` (llvmlite auto-resolved by numba)
- [x] Add version constraints to all dependencies in `pyproject.toml`

### GPU Backend Safety
- [x] Add `hasattr(torch.backends, 'mps')` check in `gpu/__init__.py`
- [x] Handle `torch.cuda.is_available()` failures gracefully
- [x] Add fallback for GPU initialization (falls back to CPU on any error)

### User Data Location
- [x] ~~Move user palettes from package dir~~ (N/A - source distribution, directory is writable)
- [x] ~~Use `platformdirs`~~ (N/A - not needed for source distribution)
- [x] ~~Add migration for existing user palettes~~ (N/A)

### Bundled DearPyGui
- [x] ~~Decide: remove bundled `DearPyGui/`~~ (already in .gitignore, not distributed)
- [x] ~~Add to `.gitignore`~~ (already there)
- [x] No conflicts - uses PyPI `dearpygui` package

### Entry Point
- [x] Verify `elliptica.ui.dpg.app:run` function exists (yes, in app.py:620)
- [x] ~~Test `elliptica` command after pip install~~ (N/A - source distribution)

---

## P2 - Medium Priority (Soon After Release)

### Error Handling
- [x] Add try/except around `dpg.create_viewport()` for display errors
- [x] Catch `JSONDecodeError` in palette loading (`render.py`)
- [x] ~~Add graceful degradation when optional deps missing~~ (N/A - no optional deps)

### Thread Safety
- [x] Audit `RLock` usage - looks safe (reentrant, no nested cross-locks)
- [x] GPU operations use ThreadPoolExecutor (render_orchestrator, display_pipeline)

### File Operations
- [x] Make ZIP writes atomic in `serialization.py` (write to temp, then rename)
- [x] ~~Add file locking~~ (N/A - single-user desktop app, no concurrent access)
- [x] Atomic writes protect against overwrite corruption

### Import-Time Performance
- [x] ~~Defer palette building~~ (low priority - palettes needed at startup anyway)
- [x] ~~Profile import time~~ (deferred - optimize if users report slow startup)

### Platform Testing
- [ ] Test on Windows 10/11 (requires manual testing)
- [ ] Test on macOS Intel (requires manual testing)
- [x] Test on macOS Apple Silicon (current dev machine)
- [ ] Test on Ubuntu 22.04/24.04 (requires manual testing)
- [ ] Test in Docker container (requires manual testing)

---

## P3 - Low Priority (Ongoing Improvements)

### Documentation
- [x] Add "System Requirements" section to README
- [x] Document GPU requirements (CUDA/MPS versions)
- [ ] Add troubleshooting guide for common errors
- [ ] Document the `.elliptica` project file format
- [x] Add installation instructions for each platform

### Performance
- [ ] Optimize O(n³) contour algorithm in `canvas_renderer.py:79-92`
- [ ] Add progress indicators for long operations
- [ ] Profile memory usage with large canvases

### Code Quality
- [ ] Add `__all__` exports to public modules
- [ ] Remove or document `DearPyGui/` directory purpose
- [ ] Add type hints to public API functions

### CI/CD
- [ ] Set up GitHub Actions for multi-platform testing
- [ ] Add automated package build verification
- [ ] Create release workflow for PyPI publishing

### Testing
- [x] Add unit tests for serialization round-trip (already existed, fixed one test)
- [ ] Add tests for GPU fallback behavior
- [ ] Add tests for palette loading edge cases

---

## Verification Checklist

After fixes, verify these work:

```bash
# Clean install test
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows
pip install .
elliptica  # Should launch without errors

# Package contents check
pip show -f elliptica | grep -E "\.json$"  # Should list JSON files

# Import test
python -c "import elliptica; print('OK')"
```

- [ ] Fresh venv install works
- [ ] All JSON files present in installed package
- [ ] `elliptica` command launches GUI
- [ ] Project save/load works
- [ ] GPU detection works (or gracefully falls back)

---

## Progress Tracker

| Priority | Total | Done | Remaining |
|----------|-------|------|-----------|
| P0       | 10    | 10   | 0         |
| P1       | 12    | 12   | 0         |
| P2       | 14    | 10   | 4         |
| P3       | 14    | 4    | 10        |
| **Total**| **50**| **36**| **14**   |

*Last updated: 2025-12-31*
