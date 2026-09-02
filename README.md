# DiffuVision AI

> **Work in Progress (WIP)** — This application is not available as an `.exe` yet.

## Setup

To run **DiffuVision AI**, execute the Python application using **Visual Studio Code** or your preferred terminal.

> **Hardware Recommendation:** An NVIDIA GPU with CUDA support is strongly recommended. CPU-only execution is currently untested and may result in significantly slower performance.

### 1. Download the Project

Clone or download this repository, then open the project folder in **Visual Studio Code**.

### 2. Download Required Models

Download one or both of the following inpainting models:

* **DreamShaper 8 Inpainting**
  * **Link:** [DreamShaper_8_INPAINTING.inpainting.safetensors](https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_8_INPAINTING.inpainting.safetensors)
  * **Best for** General/overall use (real-life, anime)

**OR**

* **LazyMix v4.0 Inpainting**
  * **Link:** [lazymixRealAmateur_v40Inpainting.safetensors](https://huggingface.co/TheImposterImposters/LazyMix-v4.0-inpainting/blob/main/lazymixRealAmateur_v40Inpainting.safetensors)
  * **Best for** Photorealistic/real-life images only

### 3. Place the Models

After downloading the models, place them in:

```text
DiffuVision AI/model/
```

The folder structure should look something like:

```text
DiffuVision AI/
├── model/
│   ├── DreamShaper_8_INPAINTING.inpainting.safetensors
│   └── lazymixRealAmateur_v40Inpainting.safetensors
├── app.py
└── ...
```

### OR If you use File Paths, use a `.env` File

Instead of placing the models inside the `model` folder, you can specify their full file paths using a `.env` file.

Create a file named:

```text
.env
```

Then add:

```env
DREAMSHAPER_MODEL="YOUR FULL PATH TO DreamShaper_8_INPAINTING.inpainting.safetensors"
LAZYMIX_MODEL="YOUR FULL PATH TO lazymixRealAmateur_v40Inpainting.safetensors"
```

Replace the placeholder paths with the actual locations of the model files on your computer.

### 4. Download SAM2

Download SAM2 (including `sam2.1_hiera_tiny.pt`), rename the folder from `sam2` to `sam2_repo`, and place it in the directory structure shown below:

```text
DiffuVision AI/
├── modules/
│   ├── sam2_repo/
│   │   ├── sam2.1_hiera_tiny.pt
│   │   └── ...
│   └── ...
├── app.py
└── ...
```

### 5. Run the Application

Open the project in **Visual Studio Code** and run:

```bash
py app.py
```

The application will download and install the required Python dependencies when needed.

## Notes & Updates

This project is actively under development and does not currently have a standalone executable (`.exe`).

* **Date Created:** September 1, 2026
* **Status:** Work in Progress (WIP)

---

### September 2, 2026

* **Improved Masking:** Enhanced cutout precision by fine-tuning the `rembg` mask threshold *(Not Final)*.
* **Added Upscaling Module:** Included `model/upscale_img.py`.
* **Global Variable Handling:** Updated `imgclass` to properly modify string values across global variables.
* **Added Face Restoration Toggle:** Added an option to enable or disable face restoration on output images *(Default: Off)*.
* **Recommended Default Configuration Example:**
  ```json
  {
      "positive_prompt": "black tuxedo, necktie",
      "negative_prompt": "wrinkled, low quality, bad anatomy, deformed, extra limbs",
      "segmentation_positive_prompt": "shirt, pants",
      "segmentation_negative_prompt": "",
      "steps": 30,
      "cfg": 7.3,
      "strength": 0.81,
      "seed": -1,
      "guidance_rescale": 0,
      "mask_blur": 0.2,
      "mask_outline_thickness": 1.3
  }
  ```