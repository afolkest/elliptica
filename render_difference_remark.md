Script (test_output_with_clahe.png):
  1. Highpass filter
  2. Downsample
  3. Normalize highpass to [0,1] ← KEY STEP
  4. CLAHE on [0,1] data
  5. Percentile clip on top of CLAHE
  6. Apply colormap

  UI pipeline:
  1. LIC → apply_highpass_clahe which does:
    - Highpass (creates negative values)
    - CLAHE on the highpass data (still has negatives!)
    - Rescale CLAHE back to highpass range
  2. Downsample
  3. Percentile clip + colorize

  The key difference: the script normalizes the highpass to [0,1] BEFORE applying CLAHE, but the UI applies CLAHE directly to the highpass data (which
  has negative values) and then rescales it back.

  So no, you cannot reproduce that exact look in the UI - the processing pipeline is different. The UI is missing the "normalize before CLAHE" step.
