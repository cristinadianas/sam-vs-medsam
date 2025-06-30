# SAM vs MedSAM – Medical Image Segmentation Benchmark

This repository provides a comprehensive benchmarking suite for the **Segment Anything Model (SAM)** and **MedSAM** on 2D slices of medical images from the Medical Segmentation Decathlon. You can run single‐case evaluations, batch experiments across multiple organs/tasks, visualize results offline, or explore them interactively in a Streamlit app.

---

## Project Structure

```
.
├── segment_anything/            # SAM model architecture, from Meta’s GitHub
│
├── checkpoints/                 # Place your model checkpoints here
│   ├── sam_vit_b_01ec64.pth
│   └── medsam_vit_b.pth
│
├── msd/                         # Root of Medical Segmentation Decathlon data
│   ├── Task01_BrainTumour/
│   │   ├── imagesTr/            # .nii.gz image volumes
│   │   └── labelsTr/            # .nii.gz ground-truth volumes
│   ├── Task02_Heart/
│   └── …                        # Other tasks up to Task10_Colon
│
├── results/                     # Benchmark outputs (CSVs with metrics, logs)
│
├── final_results/               # Post-processed summary results
│
├── figures_and_plots/           # All figures used in the report
│   ├── dsc_vs_bbox.png
│   ├── nsd_vs_bbox.png
│   ├── taskwise_dsc_bar.png
│   ├── heatmap_dsc_diff.png
│   └── example_brain_tumour.png
│
├── app.py                       # Streamlit UI: upload an image, draw a box, run SAM/MedSAM
├── benchmark.py                 # Core CLI: run SAM & MedSAM on all slices of a task
├── batch_benchmarks.py          # Loop over multiple tasks & box sizes via subprocess
├── visualize.py                 # Offline plotting: overlays of SAM, MedSAM, GT & intersections
└── requirements.txt             # Python dependencies
│
└── final_report.pdf             # Compiled paper/report (IEEE format)
```

---

## Installation

1. **Clone**  
   ```bash
   git clone https://github.com/your-username/sam-vs-medsam.git
   cd sam-vs-medsam
   ```

2. **Create & activate** a virtual environment (recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download & place** model checkpoints  
   - `checkpoints/sam_vit_b_01ec64.pth`  
   - `checkpoints/medsam_vit_b.pth`

5. **Prepare** MSD data under `./msd/TaskXX_<Name>/imagesTr` and `labelsTr` (keep directory names exact).

---

## Configuration Highlights

### 1. `benchmark.py`

This is the core benchmarking script responsible for running SAM and MedSAM on a set of 2D image slices from a specified medical task (e.g., Heart, BrainTumour). It supports both test mode (with hardcoded defaults) and CLI mode, where users can specify task name, bounding box enlargement, and sample limits. The script loads 3D volumes and labels, extracts individual slices, computes bounding box prompts from ground truth, runs segmentation with both models, and calculates evaluation metrics such as Dice and Normalized Surface Dice. Results are stored in a structured CSV file, and timings are logged to a separate file.

- **Modes**  
  - `TEST_MODE = True` (default):  
    - `TASK_NAME = "Task02_Heart"`  
    - `BBOX_ENLARGE_PIXELS = 0`  
    - `NUM_TEST_SCANS = 1`  
    - `LIMIT = None`  
  - `TEST_MODE = False`: parses CLI args  
    - `--task`  ( e.g. `Task01_BrainTumour` )  
    - `--bbox`  ( enlarge GT-box by N pixels )  
    - `--limit` ( max number of scans )

- **Metrics**  
  - **Dice coefficient**  
  - **Normalized Surface Dice** (tolerance = 1.0 px)

- **Outputs**  
  - Per-slice results → `<TASK_NAME>_Labels[LABELS]_Bbox[N]_Results.csv`  
  - Timing log → `benchmark_times.txt`

### 2. `batch_benchmarks.py`

This script serves as a batch automation driver for benchmark.py. It loops over multiple tasks (e.g., all 10 MSD tasks) and multiple bounding box sizes by spawning subprocesses. It's useful for evaluating the scalability and robustness of SAM and MedSAM across different organs and prompt settings. While it currently has hardcoded values for the tasks, box sizes, and slice limits, these can be easily customized. It's an efficient way to benchmark across datasets without manually calling the core script for each case.

- Hard-coded to run `benchmark.py` over a list of tasks & box sizes via `subprocess.run`.  
- **Edit** `venv_python` to point to your Python executable.  
- **Customize** `tasks`, `bbox_values`, `limit` as needed.

### 3. `visualize.py`

This script provides offline visualization utilities for inspecting the performance of SAM and MedSAM on a specific slice from a medical dataset. After extracting a selected slice using parameters defined at the top of the script (task name, image index, slice index, etc.), it generates bounding boxes, runs both segmentation models, and calculates evaluation metrics. It then produces a 2x3 grid of matplotlib plots showing the original image, predicted masks, and intersection with ground truth. This tool is especially helpful for developers and researchers to qualitatively understand model behavior and performance on a case-by-case basis.

- **Top‐of‐file settings**:  
  ```python
  TASK_NAME           = "Task02_Heart"
  IMAGE_INDEX         = 3          # selects file suffix “_<INDEX>.nii.gz”
  SLICE_INDEX         = 50         # z-slice within the volume
  BBOX_ENLARGE_PIXELS = 0
  ```
- **Workflow**:  
  1. Load 3D volume & GT, extract 2D slice (handles both single- and multi-modal).  
  2. Compute GT bounding box → run SAM & MedSAM.  
  3. Calculate Dice & NSD.  
  4. Plot 2×3 grid:  
     - Original (with bbox)  
     - SAM overlay  
     - MedSAM overlay  
     - SAM ∩ GT  
     - MedSAM ∩ GT

### 4. `app.py`

This script defines a Streamlit web application that allows users to interactively test SAM and MedSAM on 2D image slices. Users can upload an image (PNG or JPEG), define a bounding box by clicking two points, choose which model to run, and view the predicted segmentation overlaid on the image. The script includes utilities for loading models, generating masks from bounding boxes, and blending mask overlays visually. It’s ideal for quick experimentation, debugging, or showcasing model capabilities to non-technical stakeholders in a simple, browser-based interface.

- **Dependencies**: `streamlit`, `streamlit-image-coordinates`  
- **Checkpoints** loaded from `checkpoints/`.  
- **UI Flow**:  
  1. Upload JPG/PNG.  
  2. Click two corners → show bounding box.  
  3. Choose **SAM** or **MedSAM**, ▶️ **RUN**.  
  4. View blue mask overlay on the image.  
  5. **Refresh** to restart.

- **Key functions**:  
  - `load_predictor(ckpt_path)` — caches a `SamPredictor` on CPU/GPU.  
  - `run_segmentation(...)` — builds box, predicts single mask.  
  - `overlay_mask(...)` — composites mask in blue.  

---

## Usage Examples

### A. Single-task Benchmark

```bash
# for all slices of Heart, no box enlargement
python benchmark.py --task Task02_Heart --bbox 0 --limit 50
```

### B. Batch Benchmark

```bash
python batch_benchmarks.py
```
*(Ensure `venv_python` is correct.)*

### C. Offline Visualization

```bash
# edit settings at top of visualize.py, then:
python visualize.py
```

### D. Interactive Exploration

```bash
streamlit run app.py
```

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---
