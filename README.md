# DiffuVision AI

> **Work in Progress (WIP)** — This application is not available as an `.exe` yet.

## Setup

To run DiffuVision AI, you currently need to use **Visual Studio Code** and run the Python application manually.

### 1. Download the Project

Clone or download this repository, then open the project folder in **Visual Studio Code**.

### 2. Download the Required Models

Download the following inpainting models:

* **DreamShaper 8 Inpainting**
  https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_8_INPAINTING.inpainting.safetensors

* **LazyMix v4.0 Inpainting**
  https://huggingface.co/TheImposterImposters/LazyMix-v4.0-inpainting/blob/main/lazymixRealAmateur_v40Inpainting.safetensors

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

### Alternative: Use a `.env` File

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

### 4. Run the Application

Open the project in **Visual Studio Code** and run:

```bash
python app.py
```

The application will download and install the required Python dependencies when needed.

## Notes

The project is still under development and does not currently have a standalone `.exe` version.

**Date Created:** September 1, 2026
**Status:** WIP
