# FlowCol Tools

Utility scripts for FlowCol.

## Palette Explorer

Standalone app for rapidly exploring and testing color palettes on rendered images.

### Features

- **Real-time palette preview**: Switch between palettes with instant visual feedback
- **GPU-accelerated**: Uses MPS on Apple Silicon for <20ms updates
- **Brightness/Contrast/Gamma controls**: Fine-tune appearance without re-rendering
- **Palette browser**: Visual grid of all available palettes with colormap previews
- **Randomize button**: Quickly cycle through palettes for inspiration

### Usage

```bash
# Activate venv first
source venv/bin/activate

# Launch palette explorer with a project
python tools/palette_explorer.py projects/your_project.flowcol
```

**Requirements:**
- The project must have a cached render (`.flowcol.cache` file)
- To create a cache, open the project in FlowCol and render it at least once

### Workflow

1. Create and render a project in the main FlowCol app
2. Launch palette explorer pointing to that project
3. Click "Change Palette" to browse available palettes
4. Use sliders to adjust brightness, contrast, gamma
5. Click "🎲 Randomize" to explore random palettes
6. When you find a palette you like, note its name and apply it in the main app

### Adding Palettes

The palette explorer displays all palettes from your FlowCol library (`_RUNTIME_PALETTES`).

To add more palettes:

1. Use the main FlowCol app's palette management UI (right-click to delete)
2. Run `scripts/import_matplotlib_palettes.py` to bulk-import matplotlib colormaps
3. Create custom palettes using the API (future feature: template-based generation)

### Architecture

- **Loads cached LIC array** from `.flowcol.cache` (no physics recomputation)
- **Applies colorization on-the-fly** using GPU LUT operations
- **Minimal UI** built with DearPyGui
- **Standalone** - doesn't modify your project files

### Performance

- **Palette switch**: ~10-20ms (GPU)
- **Slider adjustment**: ~10-20ms (GPU)
- **Resolution**: Works with full render resolution (no downsampling)

### Future Enhancements

- Template-based palette generation (sunset, ocean, neon, etc.)
- Save favorite palettes directly from explorer
- Export variations as image files
- Side-by-side comparison mode
- Integration with external palette APIs
