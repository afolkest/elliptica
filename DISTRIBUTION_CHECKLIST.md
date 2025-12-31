# Elliptica Distribution Fix Checklist

Track progress on distribution issues. Check off items as they're completed.

---

## P0 - Critical (Fix Immediately)

These will break 95%+ of installations.

### Packaging & Data Files
- [ ] Add `[tool.setuptools.package-data]` to `pyproject.toml` for JSON files
- [ ] Create `MANIFEST.in` file for source distributions
- [ ] Verify JSON files included: `palettes/library.json`, `palettes_user.json`

### Python Version
- [ ] Change `requires-python` from `>=3.10` to `>=3.11` in `pyproject.toml`

### NumPy 2.x Compatibility
- [ ] Fix `render.py:732` - remove `dtype` param from `np.power()`, cast result instead
- [ ] Fix `render.py:780` - same issue, remove `dtype` param

### Serialization Safety
- [ ] Add `.get()` defaults in `serialization.py:178-187` for backward compatibility
- [ ] Create or remove reference to `elliptica.migrate` script (`serialization.py:110-115`)

### Headless Environment
- [ ] Add display/headless check before `dpg.create_context()` in `app.py`
- [ ] Provide helpful error message for headless environments

---

## P1 - High Priority (Before Any Release)

### Dependency Pinning
- [ ] Pin `numpy>=1.24,<2.0` until NumPy 2.x compat is verified
- [ ] Pin `scipy>=1.10.0,<2.0`
- [ ] Pin `numba>=0.57.0` with compatible `llvmlite`
- [ ] Add version constraints to all dependencies in `pyproject.toml`

### GPU Backend Safety
- [ ] Add `hasattr(torch.backends, 'mps')` check in `gpu/__init__.py:27`
- [ ] Handle `torch.cuda.is_available()` failures gracefully
- [ ] Add timeout/fallback for GPU initialization

### User Data Location
- [ ] Move user palettes from package dir to user config dir (`render.py:403`)
- [ ] Use `platformdirs` or `appdirs` for cross-platform user data paths
- [ ] Add migration for existing user palettes

### Bundled DearPyGui
- [ ] Decide: remove bundled `DearPyGui/` or document why it's needed
- [ ] If keeping, add to `.gitignore` or package properly
- [ ] Ensure no conflicts with PyPI `dearpygui` package

### Entry Point
- [ ] Verify `elliptica.ui.dpg.app:run` function exists and is correct
- [ ] Test `elliptica` command works after pip install

---

## P2 - Medium Priority (Soon After Release)

### Error Handling
- [ ] Add try/except around `dpg.create_viewport()` for display errors
- [ ] Catch `JSONDecodeError` in palette loading (`render.py`)
- [ ] Add graceful degradation when optional deps missing

### Thread Safety
- [ ] Audit `RLock` usage in `state.py` for deadlock potential
- [ ] Ensure GPU operations don't block UI thread

### File Operations
- [ ] Make ZIP writes atomic in `serialization.py` (write to temp, then rename)
- [ ] Add file locking for concurrent access protection
- [ ] Validate project files before overwriting

### Import-Time Performance
- [ ] Defer palette building in `render.py:491-493` (lazy load)
- [ ] Profile and optimize import time

### Platform Testing
- [ ] Test on Windows 10/11
- [ ] Test on macOS Intel
- [ ] Test on macOS Apple Silicon
- [ ] Test on Ubuntu 22.04/24.04
- [ ] Test in Docker container

---

## P3 - Low Priority (Ongoing Improvements)

### Documentation
- [ ] Add "System Requirements" section to README
- [ ] Document GPU requirements (CUDA/MPS versions)
- [ ] Add troubleshooting guide for common errors
- [ ] Document the `.elliptica` project file format
- [ ] Add installation instructions for each platform

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
- [ ] Add unit tests for serialization round-trip
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
| P0       | 10    | 0    | 10        |
| P1       | 12    | 0    | 12        |
| P2       | 14    | 0    | 14        |
| P3       | 14    | 0    | 14        |
| **Total**| **50**| **0**| **50**    |

*Last updated: 2025-12-31*
