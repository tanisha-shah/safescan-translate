# Agrifold Implementation Workflow

This document tracks everything we do, the overall flow, the tools used, and the reasons behind our decisions as we start fresh on the Agrifold implementation.

## Phase 1: Environment Setup and Validation

**Goal:** Ensure the system environment is properly set up with all required dependencies without running the complete model training pipeline.

### Why this approach?
Before diving into the core implementation or running long training pipelines, we need to guarantee that the environment (Python dependencies, PyTorch, and CUDA if available) is correctly installed. By isolating this step, we prevent complex runtime errors related to missing packages.

### Actions Taken

1. **Modified `runner.py`**:
   - **Action**: Commented out the lines executing `src.experiments.run_all_models`.
   - **Reason**: The user explicitly requested to skip the model execution step to ensure we are only validating and installing dependencies.

2. **Executed `runner.py` (First Run)**:
   - **Action**: Ran the script.
   - **Result**: The script incorrectly reported CUDA as unavailable and installed CPU PyTorch.

3. **Investigated GPU Availability**:
   - **Action**: Ran `nvidia-smi` to verify hardware.
   - **Reason**: The user suspected the CPU version was installed despite having an Nvidia GPU.
   - **Result**: Confirmed an RTX 3050 is present. The issue was a flaw in `runner.py` which checks `torch.cuda.is_available()` on the *currently* installed PyTorch. Since CPU PyTorch was previously installed, it fell back to reinstalling CPU PyTorch.

4. **Robust `runner.py` Fix (No Hardcoding)**:
   - **Action**: Rewrote the system check logic in `runner.py` to use Python's `shutil.which("nvidia-smi")` and `subprocess` to directly query the system hardware. 
   - **Reason**: This correctly detects the presence of an Nvidia GPU dynamically on any system without relying on a pre-existing (and potentially incorrect) PyTorch installation.

5. **Executed `runner.py` (Second Run)**:
   - **Action**: Executed the updated script to fetch the CUDA version of PyTorch.
   - **Result**: The script successfully detected the Nvidia GPU. However, `pip` skipped the installation because it saw that `torch-2.12.0+cpu` was already installed and considered it functionally equivalent to `torch-2.12.0+cu121` in terms of version numbering.

6. **Added `--force-reinstall`**:
   - **Action**: Modified the CUDA pip install command in `runner.py` to include the `--force-reinstall` flag.
   - **Reason**: To force pip to overwrite the existing CPU installation with the CUDA package.

7. **Executed `runner.py` (Third Run)**:
   - **Action**: Ran the script with the `--force-reinstall` flag.
   - **Result**: Successfully triggered the 2.5GB download of PyTorch with CUDA 12.1 support.

### Concept Note: PyTorch CUDA vs System CUDA
*For future reference:* Having the Nvidia Display Drivers and CUDA installed globally on your Windows system is necessary, but Python doesn't use it directly out-of-the-box. PyTorch is distributed in a unique way:
- **CPU PyTorch (`+cpu`)**: Very small, lacks GPU acceleration libraries.
- **CUDA PyTorch (`+cu121`)**: Very large (~2.5 GB) because it comes pre-bundled with its own static copies of the Nvidia math libraries (`cuDNN`, `cuBLAS`, etc.).
We weren't downloading the CUDA toolkit again; we were specifically downloading the version of PyTorch that has been compiled to talk to your already-installed GPU.

### Next Steps
- Verify the completion of the PyTorch download and proceed with the dataset verification/generation for the model pipeline.

---

## Phase 2: Redirecting Data and Launching Pipeline 1 Training

**Goal:** Run the Pipeline 1 FedAvg training using the pre-split data in `data/overlapping_augmented/` instead of the missing `data/client_raw/`.

### Background
The original Pipeline 1 flow expects:
1. Raw images in `data/client_raw/<client_id>/<class>/`
2. Pipeline auto-splits into `data/client_split/` (70/15/15)
3. Pipeline preprocesses into `data/client_processed/` (resize → crop to 224x224)
4. Training reads from `data/client_processed/`

Since `data/client_raw/` does not exist (gitignored), but `data/overlapping_augmented/` already contains pre-split and pre-processed data in the correct `client_id/train|val|test/<class>/` structure, we bypass steps 1–3.

### Actions Taken

1. **Verified `data/overlapping_augmented/` structure**:
   - Confirmed the directory has `client_1`, `client_2`, `client_3` subdirectories.
   - Each client has `train/`, `val/`, `test/` subdirectories with class folders inside (e.g., `bacterial_blight`, `cercospora_leaf_blight`, etc.).
   - Structure exactly matches what the training code expects from `CLIENT_PROCESSED_DIR`.

2. **Updated `src/config.py`**:
   - **Change**: `CLIENT_PROCESSED_DIR` redirected from `data/client_processed` → `data/overlapping_augmented`.
   - **Reason**: The training code reads from `CLIENT_PROCESSED_DIR`. Pointing it at the already-prepared data avoids any re-processing.

3. **Updated `runner.py`**:
   - **Change**: Uncommented the training lines and added `--skip-data-setup` flag to the subprocess call.
   - **Reason**: `--skip-data-setup` tells `run_all_models.py` to skip the split/preprocess steps entirely and go straight to training, since the data is already prepared.

### Execution Result: ✅ Pipeline 1 Completed Successfully

**Exit code: 0** — All 4 models trained with Flower FedAvg over 50 rounds.

Results saved to:
- `results/comparison.csv`
- `results/comparison.json`
- `results/comparison.xlsx`

---

## Pipeline 1 Results: FedAvg (50 Rounds, 3 Local Epochs, LR=3e-5)

**Framework:** Flower gRPC | **Strategy:** FedAvg | **Data:** `data/overlapping_augmented/` | **Device:** CUDA (RTX 3050)

| Model | Test Acc | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Min Class Recall | Train Time |
|---|---|---|---|---|---|---|---|
| **vit_small** | **0.9125** | **0.8720** | **0.9055** | **0.9250** | **0.8655** | 0.3750 | ~4.67 hrs |
| mobilenet_v2 | 0.6623 | 0.5059 | 0.5813 | 0.5050 | 0.5565 | 0.0000 | ~2.18 hrs |
| resnet18 | 0.5531 | 0.3818 | 0.4882 | 0.5040 | 0.4303 | 0.0000 | ~2.20 hrs |
| efficientnet_b0 | 0.4376 | 0.3805 | 0.4060 | 0.5529 | 0.4742 | 0.0408 | ~2.96 hrs |

### Key Observations
- **ViT-Small dominates** with 91.25% test accuracy and 87.2% macro F1, outperforming CNN models by a wide margin under non-IID federated settings.
- **MobileNetV2 and ResNet18** show moderate accuracy (~55–66%) but struggle with rare classes (per_class_recall_min = 0.0), meaning at least one class is being entirely missed by these models.
- **EfficientNet-B0** has the lowest accuracy (43.76%) despite having the second-longest training time (~3 hrs), suggesting it may need more rounds or a different LR schedule under this non-IID data distribution.
- The non-IID split (each client owns disjoint classes) is the core challenge — models must generalize across classes they have never seen locally.

---

## Phase 3: Custom Architecture (AgrifoldVGG)

**Goal:** Implement a custom VGG16 architecture (`AgrifoldVGG`) equipped with Efficient Channel Attention (ECA) modules to improve feature representation for the 7-class soybean disease dataset.

### Actions Taken

1. **Created `src/models/agrifold_vgg.py`**:
   - Implemented `AgrifoldVGG`, wrapping the standard `torchvision` VGG16 backbone.
   - Sliced the VGG16 feature extractor into its 5 natural convolutional blocks (ending at max-pooling layers).
   - Inserted an `ECAModule` (Efficient Channel Attention) after each of the 5 blocks.

2. **Implemented ECA Module**:
   - Built a parameter-efficient channel attention mechanism using a 1D convolution (`k=3`) across the channel dimension, avoiding fully connected layers entirely.

3. **Output Standardization**:
   - Replaced the standard VGG classifier head with a global average pooling (`AdaptiveAvgPool2d(1)`) and a single `Linear(512, 7)` layer.
   - Exposed a `get_features()` method returning `(B, 512)` embeddings, matching the expected interface (like the ResNet18 baseline) required by the pipeline.

---

## Phase 4: SCAFFOLD Pipeline Integration & Evaluation

**Goal:** Execute the Agrifold model on the SCAFFOLD federated learning pipeline and evaluate via the web dashboard.



### Actions Taken

1. **Wired Agrifold into Model Factory (`src/models/factory.py`)**:
   - Updated `create_model` to properly instantiate `AgrifoldVGG` when requested, ensuring the training pipelines and the web app can correctly load its architecture.

2. **Registered Model in Configurations**:
   - Added `"agrifold"` to `MODEL_NAMES` in both `src/config.py` and `scaffold_pipeline/config.py`.
   - Added `"agrifold"` to the argparse choices in `scaffold_pipeline/train_one.py` and `train_all.py`.

3. **Executed SCAFFOLD Training**:
   - Ran `python -m scaffold_pipeline.train_one --model agrifold --rounds 20 --epochs 5`.
   - *Result*: Simulates 1 server and 3 clients running the SCAFFOLD strategy sequentially on the overlapping dataset.

4. **Web Dashboard Evaluation**:
   - Executed `start_webapp.bat`.
   - *Result*: The newly trained global model and its metrics are now available for inference in the web UI.

### Execution and Dashboard Monitoring Details

- **Training Duration & Intermediate Results**: VGG16 with Efficient Channel Attention is a computationally heavy architecture. The executed command runs for 20 global rounds, where each round consists of 3 clients performing 5 local epochs. This results in processing approximately 40,000 images per round, taking an estimated 1-2 hours to complete on the current hardware. Intermediate metrics (e.g., validation loss, accuracy, and per-class F1 scores) are continuously logged in real-time to the terminal and recorded per-round in `scaffold_pipeline/results/agrifold/round_history.csv`.
- **Dashboard Synchronization**: The web application is designed to load fully finalized global models. As a result, the `AgrifoldVGG` inference and evaluation metrics will populate in the web UI only after the training script completes (or triggers early stopping) and exports the final `global_model.pth` and `metrics.json` files.

### Final Results

The SCAFFOLD training successfully completed. While the pipeline was configured for 20 rounds, the built-in **Early Stopping** mechanism was triggered at **Round 16**. The model correctly identified that the validation loss had stopped improving after Round 10, thus halting the process to prevent overfitting and save computational resources.

**Final AgrifoldVGG Metrics (Round 16):**
- **Validation Accuracy**: ~96.8%
- **Macro F1-Score**: ~96.8%

The finalized model and `metrics.json` are now permanently saved and fully integrated into the Web Dashboard for inference testing.

---

## Phase 5: Dashboard Visualization Enhancements

**Goal:** Fulfill the advanced evaluation requirements presented in the project slides, specifically adding Macro F1 convergence curves and Per-Class Radar charts to the Web UI—all without destroying the existing layout.

### Actions Taken

1. **Per-Class F1-Score (Radar Chart)**:
   - Added `Chart.js` to the dashboard via CDN.
   - Created a dedicated **🕸️ Per-Class Radar** tab to isolate the chart and prevent cluttering the main Comparison UI.
   - The radar chart automatically plots a polygon of the F1-scores for all 7 classes. This is critical for revealing imbalanced learning (e.g., high global accuracy but failing completely on a specific disease like *Downey Mildew*).

2. **Macro F1-Score Convergence Curve**:
   - Updated `app.py` to parse the `round_history.csv` and extract `val_f1_macro` per round.
   - Modified the Training Curves canvas in `index.html` to include a sleek UI Toggle switch. By default, it preserves the original Global Accuracy view, but users can now toggle it to plot Macro F1-Score over rounds to properly evaluate performance on the imbalanced dataset.

---

## Summary of File Modifications

The following table summarizes the files that were created or updated to implement the Agrifold architecture, SCAFFOLD pipeline integration, and Dashboard enhancements.

| Filename | Modification Details |
| :--- | :--- |
| `src/models/agrifold_vgg.py` | **Created**. Implemented the custom `AgrifoldVGG` architecture featuring the 1D Efficient Channel Attention (`ECAModule`) blocks. |
| `src/models/factory.py` | **Updated**. Added the `agrifold` routing into the `create_model` function so the entire pipeline can load it natively. |
| `src/config.py` | **Updated**. Registered `"agrifold"` to the global `MODEL_NAMES` list. |
| `scaffold_pipeline/config.py` | **Updated**. Added `"agrifold"` to the local `MODEL_NAMES` list used specifically by the SCAFFOLD federated scripts. |
| `scaffold_pipeline/models.py` | **Updated**. Wired the `agrifold` loader into the `get_model` utility for the federated server/clients. |
| `scaffold_pipeline/train_one.py` | **Updated**. Modified the evaluation logging step to accurately calculate and record the `val_f1_macro` in `round_history.csv`. |
| `app/app.py` | **Updated**. Adjusted the `/api/training-history` backend route to dynamically parse and return the `val_f1_macro` convergence data to the frontend. |
| `app/templates/index.html` | **Updated**. Added the `Chart.js` integration, created the new **🕸️ Per-Class Radar** tab, and added the toggle switch for the Training Curves (Accuracy vs. F1-Score). |
| `agrifold_Implementation.md` | **Created/Updated**. Wrote the detailed documentation logging the architecture, pipeline integration, and dashboard UI upgrades. |

