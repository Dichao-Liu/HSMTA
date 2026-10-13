# HSMTA

This repository provides the PyTorch implementation of **Hierarchical Spatially Merged Token Averager (HSMTA)**, presented in the manuscript **"The Simpler, the Better: A Lightweight Hierarchical Token Averager for Single-Label Microscopic Specimen Image Classification"**, accepted for publication in *IEEE Transactions on Biomedical Engineering*.

HSMTA progressively reorganizes and merges local spatial elements into hierarchical morphology descriptors, followed by simple token averaging for image-level classification. The repository also includes **HSMTA-SRR**, an SRR-inspired sparsity-regularized variant used to analyze the formation of intermediate representations.

### Figure 1: Accuracy vs. computational cost

Comparison of classification accuracy and GMACs on the DSCC dataset.

![Accuracy versus GMACs on DSCC](./accuracy_vs_gmacs.png)

### Figure 2: Overview of the proposed approach

Comparison of (a) conventional CNN processing, (b) token-mixer-based processing, and (c) the proposed hierarchical spatial merging and token averaging.

![Overview of HSMTA](./structure.png)

### Environment and Dependencies

The following is the recorded environment of the `torch_1.11` Conda environment on the Ubuntu machine used for this code. The versions were obtained from the environment report and package lists rather than inferred from the source files.

**System and hardware**

- Operating system: Ubuntu 22.04.5 LTS (Linux kernel 6.8.0-138-generic, x86_64)
- CPU: Intel Core i9-10900X @ 3.70 GHz (20 logical CPUs)
- System memory: 62 GiB
- GPUs: 2 × NVIDIA GeForce RTX 3090 Ti (24 GB nominal VRAM per GPU; 23.55 GiB reported by PyTorch)
- NVIDIA driver: 580.178.04
- CUDA version used to build PyTorch: 11.3
- cuDNN: 8.2.0
- CUDA version reported by `nvidia-smi` (driver capability): 13.0
- Standalone CUDA compiler (`nvcc`): not available in the recorded shell environment

**Python and main package versions**

| Package | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch (`torch`) | 1.11.0 |
| `torchvision` | 0.12.0 |
| `torchaudio` | 0.11.0 |
| `timm` | 1.0.30 |
| `numpy` | 1.23.5 |
| `scipy` | 1.10.1 |
| `pandas` | 2.0.3 |
| `scikit-learn` | 1.3.2 |
| `scikit-image` | 0.21.0 |
| `opencv-python` | 5.0.0.93 |
| `matplotlib` | 3.7.5 |
| `Pillow` | 10.4.0 |
| `tqdm` | 4.70.1 |
| `kmeans-pytorch` | 0.3 |
| `bagnets` | 0.1 (local editable installation) |

The provided training script requires a CUDA-enabled PyTorch installation. The installed PyTorch build uses **CUDA 11.3**; the **CUDA 13.0** displayed by `nvidia-smi` reflects driver support, not the CUDA version used to build PyTorch.

**Dependency note:** `bagnets.pytorchnet` is imported by the training script, including when training HSMTA. In the recorded environment, `bagnets` was installed in editable mode from a local source tree; it is **not** a self-contained package in this repository. The corresponding module must be available to run the script. Likewise, the machine's raw `pip freeze` contains local Conda build URLs and an absolute editable-install path, so it should not be used directly as a portable `pip install -r` file.

### Source Code

The repository includes two implementations of HSMTA:

- **`hsmta_reference.py`** — our direct reference implementation following the hierarchical spatial merging and linear-projection operations described in the manuscript. The explicit patch and linear operations make it relatively slow to execute.
- **`hsmta.py`** — our more efficient convolution-based realization of the same HSMTA architecture. The corresponding linear/spatial projections are implemented as equivalent convolutional operations, reducing the execution overhead of the reference implementation. **This is the recommended implementation for training.**

The convolution-based implementation changes how the operations are computed, not the intended architecture in the manuscript. The repository also includes `hsmta_srr.py` for HSMTA-SRR, `my_utils.py` for supporting utilities, and `train_models_DSCC_5folds.py` for DSCC training and evaluation.

```text
HSMTA/
├── README.md
├── hsmta.py
├── hsmta_reference.py
├── hsmta_srr.py
├── my_utils.py
├── train_models_DSCC_5folds.py
├── accuracy_vs_gmacs.png
└── structure.png
```

The Python module filenames must match the imports in the training script (`hsmta.py` and `hsmta_reference.py`). Both figure files should be placed in the repository root with the filenames shown above so they render directly in the GitHub README.

### Dataset

The manuscript evaluates **DSCC**, **SIPaKMeD**, **C-NMC**, and **DeFungi**. The training script currently provided in this repository is for the **DSCC five-fold experiment**.

Download and prepare DSCC separately. From the repository root, arrange each fold as follows (retaining the `splited` directory name expected by the code):

```text
datasets/
└── DSCC/
    └── splited/
        └── seed_0_5fold/
            ├── 0/
            │   ├── train/
            │   │   ├── class_1/
            │   │   ├── class_2/
            │   │   └── class_3/
            │   └── validation/
            │       ├── class_1/
            │       ├── class_2/
            │       └── class_3/
            ├── 1/
            ├── 2/
            ├── 3/
            └── 4/
```

Each class directory should contain the corresponding images. Keep the class names and their alphabetical ordering consistent across all folds; the training script uses `torchvision.datasets.ImageFolder`. Dataset images and fold assignments are not included in this repository.

### Train

Run the commands from the repository root after preparing the dataset and dependencies.

**HSMTA (convolution-based implementation, recommended), DSCC fold 0:**

```bash
python train_models_DSCC_5folds.py --m hm --fold 0
```

**HSMTA (direct reference implementation, slower), DSCC fold 0:**

```bash
python train_models_DSCC_5folds.py --m hmr --fold 0
```

**HSMTA-SRR, DSCC fold 0:**

```bash
python train_models_DSCC_5folds.py --m hms --fold 0
```

Repeat with `--fold 0` through `--fold 4` for the five DSCC folds.

The supplied training script uses 500 epochs, SGD with momentum 0.9, weight decay `5e-4`, batch size 16, initial learning rate 0.002, and cosine learning-rate annealing. HSMTA, HSMTA-SRR, and the reference model use a 228 × 228 input. The current script selects GPU 0 with `CUDA_VISIBLE_DEVICES="0"`.

### Evaluation and Outputs

The DSCC training script evaluates models using the relevant fold's `validation` directory. It saves model checkpoints and training/evaluation logs under `Results_DSCC/`, a relative output directory.

The supplied files do not include pretrained checkpoints or separate training scripts for SIPaKMeD, C-NMC, and DeFungi.
