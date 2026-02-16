# Labelis Render-Only (Napari) – standalone reconstruction + TIFF export

This mini-app is a *render-only* front-end that reuses the Labelis backend:
- robust CSV/TSV localization loading (Labelis `load_localizations`)
- super-resolution reconstruction rendering (Labelis `render_dispatch`)
- exports the rendered reconstruction as TIFF (optionally with OME metadata)

## Run (dev)
From a Python environment where `labelis`, `napari`, and `PyQt5` are installed:

    python render_only_app.py

## CLI (headless)
    python render_only_cli.py --input my.csv --output out.tif --px-nm 10 --engine numba

## Notes
- By default, XY are shifted so the minimum X/Y become 0 (same as Labelis GUI/pipeline),
  which keeps image sizes reasonable if your coordinates are large.
- TIFF export stores the pixel size in OME metadata (PhysicalSizeX/Y in µm) and also writes a
  small JSON sidecar with the render settings.
