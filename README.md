# Labelis (Napari-first ROI)

## What this is
Labelis is a **Napari-first** NPC analysis application.

Hard rule enforced:
- **ROI is drawn in Napari on the rendered image BEFORE any NPC processing can run.**
- After ROI is confirmed, the user unlocks the analysis parameters (segmentation/filtering/alignment) and can run the pipeline.

## Install (recommended: conda-forge)
Create an isolated environment (Windows / Anaconda Prompt):

```bat
conda env create -f env/labelis.yml
conda activate labelis
```

## Run
From the folder containing `labelis/`:

```bat
python -m labelis
```

## Workflow
1. In the Napari dock widget:
   - Select localization CSV/TSV
   - Select output directory
   - Choose render pixel size + compute engine
   - Click **Load + Render**
2. Draw a polygon ROI in the **ROI** Shapes layer.
3. Click **Confirm ROI** (analysis controls unlock).
4. Set analysis parameters, then click **Run analysis**.

## Outputs (written inside your chosen output directory)
The run creates a MATLAB-like set of artifacts:

- `labelis_config.json` – full parameter set used
- `labelis_summary.json` – run summary (counts, p_label, derived auto-parameters)
- `labelis_npcs_final.csv` – per-NPC table
- `labelis_workspace.h5` – **workspace-style** HDF5 file (environment + parameters + ROI + tables + key arrays/images)
- `labelis_debug.log` – staged debug log containing `DEBUG(0)...DEBUG(8)`
- `labelis_final_localizations.csv` – localizations used for the final NPCs

Figures (saved as PNG):

- `figure_ELE.png`
- `figure_info_NPC.png`
- `figure_NPC_image.png`
- `figure_NPC_image_intensity_profile.png`

Additional diagnostic images (optional, controlled by `save_intermediate_images`) are also saved.
