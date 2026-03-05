# PyTorch U-Net: Retinal Blood Vessel Segmentation

### Overview
This repository contains a complete, end-to-end PyTorch implementation of the U-Net architecture built from scratch. The model is designed to perform highly precise semantic segmentation on medical imagery, specifically extracting the complex network of blood vessels from raw photographs of the human retina.

### The Dataset
This project uses the **DRIVE** (Digital Retinal Images for Vessel Extraction) dataset.
* **Inputs:** High-resolution `.tif` photographs of the inside of the eye.
* **Ground Truth:** `.gif` binary masks manually drawn by human medical experts.
* **Challenge:** The dataset contains mixed image formats and varying directory structures. This pipeline includes a custom PyTorch `Dataset` class and dynamic `os.walk()` path resolution to handle case-insensitive file routing seamlessly.

### Architecture Details
The model is a standard **Fully Convolutional Network (FCN)** utilizing the symmetrical U-Net topology:
1. **Encoder (Downsampling):** Extracts deep semantic features using $3 \times 3$ Double Convolutions and $2 \times 2$ Max Pooling.
2. **Bottleneck:** Holds the most compressed spatial representations.
3. **Decoder (Upsampling):** Reconstructs the image using Transposed Convolutions.
4. **Skip Connections:** High-resolution feature maps from the Encoder are concatenated directly into the Decoder to preserve perfect pixel-level boundary data for the tiny blood vessels.
5. **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) optimized with Adam.

### Installation & Setup
Clone the repository to your local machine:
```bash
git clone [https://github.com/Uthmans-7/unet-drive-segmentation.git](https://github.com/Uthmans-7/unet-drive-segmentation.git)
cd unet-drive-segmentation