# Elliptica Manual Testing Protocol

Quick manual tests to verify core functionality. Run after major changes.

## Startup (~1 min)

- [ ] Run `python -m elliptica.ui.dpg`
- [ ] App window appears without errors
- [ ] Terminal shows GPU backend: `GPU acceleration: MPS` or `CUDA` or `CPU`
- [ ] Demo boundary (circle) visible on canvas

## Project Save/Load (~2 min)

- [ ] Click "Load Boundary..." and import any PNG
- [ ] Drag boundary to new position
- [ ] Adjust voltage slider
- [ ] File > Save Project > save as `test_project.elliptica`
- [ ] File > New Project (canvas clears)
- [ ] File > Load Project > open `test_project.elliptica`
- [ ] Verify: boundary position and voltage restored
- [ ] Save again to same file (overwrites without error)

## Rendering (~2 min)

- [ ] Click "Render Field"
- [ ] Progress modal appears, completes without error
- [ ] LIC visualization displayed
- [ ] Adjust brightness/contrast/gamma sliders - updates live
- [ ] Change palette dropdown - colors update
- [ ] Click "Back to Edit" - returns to edit mode

## Export (~1 min)

- [ ] Render a field first (if not already)
- [ ] Export > Export Image
- [ ] Choose location, save as PNG
- [ ] Open exported file - image looks correct

## Error Handling (~1 min)

- [ ] Create a text file, rename to `fake.elliptica`
- [ ] Try File > Load Project > open `fake.elliptica`
- [ ] Error message appears (not a crash)
- [ ] App still functional after error

## Edge Cases (optional)

- [ ] Import very large PNG (2000x2000+) as boundary
- [ ] Create 5+ boundaries, save/load project
- [ ] Render at 2x or higher multiplier
- [ ] Change canvas size via "Change..." button

---

## Platform-Specific Tests

### macOS Apple Silicon
- [ ] GPU shows as `MPS`
- [ ] Rendering is fast (< 5 sec for default canvas)

### macOS Intel
- [ ] GPU shows as `CPU` (no MPS on Intel)
- [ ] App still works, just slower

### Windows
- [ ] App launches from Command Prompt or PowerShell
- [ ] File dialogs work correctly
- [ ] GPU shows as `CUDA` (if NVIDIA) or `CPU`

### Linux
- [ ] App launches from terminal
- [ ] Requires X11 or Wayland display
- [ ] GPU shows as `CUDA` (if NVIDIA) or `CPU`

---

## After Testing

Clean up test files:
```bash
rm test_project.elliptica fake.elliptica
```
