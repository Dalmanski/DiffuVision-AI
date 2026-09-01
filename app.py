import os,sys,json,time,threading,urllib.request,gc
from pathlib import Path
from tkinter import filedialog,messagebox
import customtkinter as ctk
from PIL import Image,ImageTk,ImageFilter,ImageOps
import numpy as np,torch,cv2
import torchvision.transforms.functional as TF
sys.modules.setdefault("torchvision.transforms.functional_tensor",TF)
from diffusers import StableDiffusionInpaintPipeline,DPMSolverMultistepScheduler
import modules.segdinosam2 as segdinosam2
import modules.imgclass as imgclass
import modules.gender as gender
try:
    gender.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except Exception:
    RRDBNet=RealESRGANer=None
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"model"
DEFAULT_JSON=BASE_DIR/"data/default.json"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LAZYMIX_FILENAME="lazymixRealAmateur_v40Inpainting.safetensors"
DREAMSHAPER_FILENAME="DreamShaper_8_INPAINTING.inpainting.safetensors"


def read_env_file():
    env_path=BASE_DIR/".env"
    values={}
    if not env_path.exists():
        return values
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key,value=line.split("=",1)
            elif ":" in line:
                key,value=line.split(":",1)
            else:
                continue
            values[key.strip()]=value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return values


def resolve_model_path(env_key,default_filename=None):
    value=read_env_file().get(env_key,"").strip()
    candidates=[]

    if value:
        candidates.append(value)

    if default_filename:
        candidates.append(default_filename)
        candidates.append(str(MODEL_DIR/default_filename))

    seen=set()
    for raw in candidates:
        if not raw:
            continue

        path=Path(raw)
        if not path.is_absolute():
            if default_filename and path.name.lower()==default_filename.lower() and path.parent in (Path("."), Path("")):
                path=MODEL_DIR/path.name
            else:
                path=BASE_DIR/path

        resolved=path.resolve(strict=False)
        if str(resolved) not in seen:
            seen.add(str(resolved))
            if resolved.exists():
                return resolved

    if default_filename:
        fallback=MODEL_DIR/default_filename
        if fallback.exists():
            return fallback

    return None


LAZYMIX_MODEL=resolve_model_path("LAZYMIX_MODEL",LAZYMIX_FILENAME)
DREAMSHAPER_MODEL=resolve_model_path("DREAMSHAPER_MODEL",DREAMSHAPER_FILENAME)
MODEL_OPTIONS={
    "LazyMixRealAmateur v4.0 Inpainting":str(LAZYMIX_MODEL) if LAZYMIX_MODEL else "",
    "DreamShaper 8 Inpainting":str(DREAMSHAPER_MODEL) if DREAMSHAPER_MODEL else ""
}
DEFAULT_MODEL="DreamShaper 8 Inpainting"
MAX_SIDE=768
RESIZE_TARGET=512
OUTPUT_TARGET=1080
IMAGE_CLASSES=["real life","anime","3D","cartoon"]
ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESRGAN_PATH=BASE_DIR/"weights"/"RealESRGAN_x4plus.pth"
FACE_RESTORE_POSITIVE_PROMPT="face, hair"
FACE_RESTORE_NEGATIVE_PROMPT="neck"

class ConsoleRedirect:
    def __init__(self,app):
        self.app=app

    def write(self,text):
        if not text:
            return
        if "\r" in text:
            parts=text.split("\r")
            if len(parts)>1:
                text=parts[-1]
            self.app.console_log(text,live=True)
        else:
            self.app.console_log(text)

    def flush(self):
        pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DiffuVision AI - Inpainting with Stable Diffusion")
        self.iconbitmap("favicon.ico")
        self.geometry("1600x1050")
        self.minsize(1100,760)
        self.pipe=None
        self.current_model_name=None
        self.current_model_id=None
        self.pipeline_dtype=None
        self.esrgan=None
        self.segmentation=None
        self.original_image=None
        self.input_image=None
        self.output_image=None
        self.input_path=None
        self.mask_image=None
        self.classification_result=None
        self.gender_result=None
        self.classification_loading=False
        self.processing=False
        self.models_ready=False
        self.model_loading=False
        self.segmentation_loading=False
        self.start_time=0
        self.save_job=None
        self.input_photo=None
        self.output_photo=None
        self.resize_var=ctk.BooleanVar(value=True)
        self.esrgan_input_var=ctk.BooleanVar(value=False)
        self.esrgan_output_var=ctk.BooleanVar(value=False)
        self.mask_var=ctk.BooleanVar(value=True)
        self.autosave_var=ctk.BooleanVar(value=True)
        self.model_var=ctk.StringVar(value=DEFAULT_MODEL)
        self.image_class_var=ctk.StringVar(value="real life")
        self.gender_var=ctk.StringVar(value="neutral")
        self.config_files=[]
        self.active_config_name="data/default.json"
        self.active_config_path=DEFAULT_JSON
        self.config={}
        self.env_path=BASE_DIR/".env"
        self.load_env_settings()
        self.protocol("WM_DELETE_WINDOW",self.destroy)
        self.ui()
        self.stdout_redirect=ConsoleRedirect(self)
        self.stderr_redirect=ConsoleRedirect(self)
        sys.stdout=self.stdout_redirect
        sys.stderr=self.stderr_redirect
        self.after(100,self.maximize)
        self.refresh_config_files()
        self.load_startup_config()
        threading.Thread(target=self.load_models,args=(self.model_var.get(),),daemon=True).start()

    def maximize(self):
        try:
            self.state("zoomed")
        except Exception:
            pass

    def ui(self):
        self.grid_rowconfigure(0,weight=1)
        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=1)

        self.left_frame=ctk.CTkScrollableFrame(self)
        self.left_frame.grid(row=0,column=0,sticky="nsew",padx=(10,5),pady=10)
        self.left_frame.grid_columnconfigure(0,weight=0)
        self.left_frame.grid_columnconfigure(1,weight=1)

        self.input_label=ctk.CTkLabel(self.left_frame,text="INPUT IMAGE")
        self.input_label.grid(row=0,column=0,sticky="w",padx=4,pady=(0,4))

        self.mask_btn=ctk.CTkButton(
            self.left_frame,
            text="HIDE MASK",
            command=self.toggle_mask,
            width=120,
            height=34
        )
        self.mask_btn.grid(row=0,column=1,sticky="e",padx=2,pady=(0,4))

        self.input_canvas=ctk.CTkCanvas(
            self.left_frame,
            bg="#181818",
            highlightthickness=0,
            height=430
        )
        self.input_canvas.grid(row=1,column=0,columnspan=2,sticky="ew",padx=2,pady=(0,8))

        self.upload_btn=ctk.CTkButton(
            self.left_frame,
            text="UPLOAD IMAGE",
            command=self.upload,
            height=40
        )
        self.upload_btn.grid(row=2,column=0,columnspan=2,sticky="ew",padx=2,pady=(0,10))

        ctk.CTkLabel(
            self.left_frame,
            text="MODEL:",
            anchor="w",
            width=100
        ).grid(row=3,column=0,sticky="w",padx=(4,8),pady=(0,8))

        self.model_menu=ctk.CTkOptionMenu(
            self.left_frame,
            variable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            command=self.model_changed
        )
        self.model_menu.grid(row=3,column=1,sticky="ew",padx=2,pady=(0,8))

        ctk.CTkLabel(
            self.left_frame,
            text="IMAGE CLASS:",
            anchor="w",
            width=100
        ).grid(row=4,column=0,sticky="w",padx=(4,8),pady=(0,8))

        self.image_class_menu=ctk.CTkOptionMenu(
            self.left_frame,
            variable=self.image_class_var,
            values=IMAGE_CLASSES
        )
        self.image_class_menu.grid(row=4,column=1,sticky="ew",padx=2,pady=(0,8))

        ctk.CTkLabel(
            self.left_frame,
            text="GENDER:",
            anchor="w",
            width=100
        ).grid(row=5,column=0,sticky="w",padx=(4,8),pady=(0,8))

        self.gender_menu=ctk.CTkOptionMenu(
            self.left_frame,
            variable=self.gender_var,
            values=["male","female","neutral"]
        )
        self.gender_menu.grid(row=5,column=1,sticky="ew",padx=2,pady=(0,8))

        ctk.CTkLabel(
            self.left_frame,
            text="JSON CONFIG:",
            anchor="w",
            width=100
        ).grid(row=6,column=0,sticky="w",padx=(4,8),pady=(0,8))

        self.config_menu=ctk.CTkOptionMenu(
            self.left_frame,
            values=[],
            command=self.config_changed
        )
        self.config_menu.grid(row=6,column=1,sticky="ew",padx=2,pady=(0,8))

        self.autosave_btn=ctk.CTkButton(
            self.left_frame,
            text="AUTOSAVE: ON",
            command=self.toggle_autosave,
            height=38,
            width=150
        )
        self.autosave_btn.grid(row=7,column=1,sticky="e",padx=2,pady=(0,8))

        self.json_box=ctk.CTkTextbox(
            self.left_frame,
            height=210,
            wrap="none"
        )
        self.json_box.grid(row=8,column=0,columnspan=2,sticky="ew",padx=2,pady=(0,8))
        self.json_box.bind("<KeyRelease>",self.json_changed)

        self.resize_cb=ctk.CTkCheckBox(
            self.left_frame,
            text="Auto resize original image for recommended SD inpainting",
            variable=self.resize_var
        )
        self.resize_cb.grid(row=9,column=0,columnspan=2,sticky="w",padx=4,pady=3)

        self.esrgan_input_cb=ctk.CTkCheckBox(
            self.left_frame,
            text="Enhance original image using Real-ESRGAN",
            variable=self.esrgan_input_var
        )
        self.esrgan_input_cb.grid(row=10,column=0,columnspan=2,sticky="w",padx=4,pady=3)

        self.esrgan_output_cb=ctk.CTkCheckBox(
            self.left_frame,
            text="Enhance output image using Real-ESRGAN",
            variable=self.esrgan_output_var
        )
        self.esrgan_output_cb.grid(row=11,column=0,columnspan=2,sticky="w",padx=4,pady=3)

        self.generate_btn=ctk.CTkButton(
            self.left_frame,
            text="GENERATE",
            command=self.generate,
            state="disabled",
            height=42
        )
        self.generate_btn.grid(row=12,column=0,columnspan=2,sticky="ew",padx=2,pady=(12,8))

        self.right_frame=ctk.CTkFrame(self)
        self.right_frame.grid(row=0,column=1,sticky="nsew",padx=(5,10),pady=10)
        self.right_frame.grid_rowconfigure(0,weight=1)
        self.right_frame.grid_columnconfigure(0,weight=1)

        self.output_canvas=ctk.CTkCanvas(
            self.right_frame,
            bg="#181818",
            highlightthickness=0
        )
        self.output_canvas.grid(row=0,column=0,sticky="nsew",padx=10,pady=(10,6))

        self.console=ctk.CTkTextbox(
            self.right_frame,
            height=260,
            wrap="none",
            font=("Consolas",12),
            fg_color="#0d0d0d",
            text_color="#d0d0d0"
        )
        self.console.grid(row=1,column=0,sticky="ew",padx=10,pady=6)
        self.console.configure(state="disabled")

        self.save_btn=ctk.CTkButton(
            self.right_frame,
            text="SAVE IMAGE AS",
            command=self.save,
            state="disabled",
            height=40
        )
        self.save_btn.grid(row=2,column=0,sticky="ew",padx=10,pady=(6,10))

        self.input_canvas.bind(
            "<Configure>",
            lambda e:self.show_input()
        )
        self.output_canvas.bind(
            "<Configure>",
            lambda e:self.show_output()
        )

        self.update_autosave_button()

    def toggle_mask(self):
        self.mask_var.set(not self.mask_var.get())
        self.mask_btn.configure(
            text="HIDE MASK" if self.mask_var.get() else "SHOW MASK"
        )
        self.show_input()

    def load_env_settings(self):
        values={}
        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding="utf-8").splitlines():
                    line=line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key,value=line.split(":",1)
                    values[key.strip()]=value.strip().strip('"').strip("'")
            except Exception:
                values={}

        global LAZYMIX_MODEL,DREAMSHAPER_MODEL,MODEL_OPTIONS
        env_values=read_env_file()
        LAZYMIX_MODEL=resolve_model_path("LAZYMIX_MODEL",LAZYMIX_FILENAME)
        DREAMSHAPER_MODEL=resolve_model_path("DREAMSHAPER_MODEL",DREAMSHAPER_FILENAME)
        MODEL_OPTIONS={
            "LazyMixRealAmateur v4.0 Inpainting":str(LAZYMIX_MODEL) if LAZYMIX_MODEL else "",
            "DreamShaper 8 Inpainting":str(DREAMSHAPER_MODEL) if DREAMSHAPER_MODEL else ""
        }

        config_value=values.get("JSON_config","").replace("\\","/")
        autosave_value=values.get("JSON_autosave",None)

        if config_value:
            candidate=Path(config_value)
            if not candidate.is_absolute():
                candidate=BASE_DIR/candidate
            self.startup_config_path=candidate
        else:
            self.startup_config_path=DEFAULT_JSON

        if autosave_value is None:
            self.autosave_var.set(True)
        else:
            self.autosave_var.set(
                autosave_value.strip().lower() in ("1","true","yes","on")
            )

    def write_env_settings(self):
        values={}

        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding="utf-8").splitlines():
                    line=line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    key,value=line.split(":",1)
                    values[key.strip()]=value.strip().strip('"').strip("'")
            except Exception:
                pass

        values["JSON_config"]=self.relative_config_path()
        values["JSON_autosave"]="True" if self.autosave_var.get() else "False"

        lines=[]
        written=set()

        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding="utf-8").splitlines():
                    stripped=line.strip()

                    if not stripped or stripped.startswith("#") or ":" not in stripped:
                        lines.append(line)
                        continue

                    key=stripped.split(":",1)[0].strip()

                    if key=="JSON_config":
                        lines.append(
                            f'JSON_config: "{values["JSON_config"]}"'
                        )
                        written.add(key)

                    elif key=="JSON_autosave":
                        lines.append(
                            f'JSON_autosave: {values["JSON_autosave"]}'
                        )
                        written.add(key)

                    else:
                        lines.append(line)

            except Exception:
                lines=[]

        if "JSON_config" not in written:
            lines.append(
                f'JSON_config: "{values["JSON_config"]}"'
            )

        if "JSON_autosave" not in written:
            lines.append(
                f'JSON_autosave: {values["JSON_autosave"]}'
            )

        self.env_path.write_text(
            "\n".join(lines)+"\n",
            encoding="utf-8"
        )

    def relative_config_path(self):
        try:
            return self.active_config_path.relative_to(BASE_DIR).as_posix()
        except Exception:
            return str(self.active_config_path).replace("\\","/")

    def relative_display_path(self,path):
        try:
            return path.relative_to(BASE_DIR).as_posix()
        except Exception:
            return str(path).replace("\\","/")

    def refresh_config_files(self):
        data_dir=BASE_DIR/"data"
        data_dir.mkdir(parents=True,exist_ok=True)

        self.config_files=sorted(
            [p for p in data_dir.glob("*.json") if p.is_file()],
            key=lambda p:p.name.lower()
        )

        values=[
            self.relative_display_path(p)
            for p in self.config_files
        ]

        self.config_menu.configure(
            values=values if values else ["data/default.json"]
        )

    def load_startup_config(self):
        self.refresh_config_files()

        candidates=self.config_files.copy()
        target=self.startup_config_path

        if not target.exists() or target.suffix.lower()!=".json":
            target=(
                DEFAULT_JSON
                if DEFAULT_JSON.exists()
                else (candidates[0] if candidates else target)
            )

        if target not in candidates and target.exists():
            candidates.append(target)
            candidates=sorted(
                candidates,
                key=lambda p:p.name.lower()
            )
            self.config_files=candidates
            self.config_menu.configure(
                values=[
                    self.relative_display_path(p)
                    for p in candidates
                ]
            )

        if target.exists():
            self.load_config(target)
            self.config_menu.set(
                self.relative_display_path(target)
            )

        self.update_autosave_button()
        self.write_env_settings()

    def config_changed(self,choice):
        if (
            self.processing
            or self.model_loading
            or self.segmentation_loading
            or self.classification_loading
        ):
            return

        target=BASE_DIR/choice

        if not target.exists():
            return

        try:
            self.load_config(target)
            self.write_env_settings()
        except Exception as e:
            messagebox.showerror(
                "Configuration Error",
                str(e)
            )

    def toggle_autosave(self):
        if self.processing:
            return

        self.autosave_var.set(
            not self.autosave_var.get()
        )

        self.update_autosave_button()
        self.write_env_settings()

        if self.autosave_var.get():
            self.save_json()

    def update_autosave_button(self):
        if not hasattr(self,"autosave_btn"):
            return

        if self.autosave_var.get():
            self.autosave_btn.configure(
                text="AUTOSAVE: ON",
                fg_color="#1f8f3a",
                hover_color="#176b2c"
            )
        else:
            self.autosave_btn.configure(
                text="AUTOSAVE: OFF",
                fg_color="#666666",
                hover_color="#555555"
            )

    def console_log(self,text,color=None,live=False):
        if text is None:
            return

        text=str(text)
        text=text.replace("\x1b[K","")
        text=text.replace("\x1b[2K","")
        text=text.strip("\n")

        if not text:
            return

        def update():
            try:
                self.console.configure(state="normal")

                if live:
                    line_start=self.console.index(
                        "end-1c linestart"
                    )
                    self.console.delete(
                        line_start,
                        "end-1c"
                    )
                    self.console.insert(
                        "end",
                        text
                    )
                else:
                    self.console.insert(
                        "end",
                        text
                    )

                    if not text.endswith("\n"):
                        self.console.insert(
                            "end",
                            "\n"
                        )

                self.console.see("end")
                self.console.configure(state="disabled")

            except Exception:
                pass

        try:
            self.after(0,update)
        except Exception:
            pass

    def cleanup_gpu(self):
        try:
            if DEVICE=="cuda":
                torch.cuda.synchronize()
        except Exception:
            pass

        gc.collect()

        if DEVICE=="cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def save_current_config(self):
        if not self.active_config_path:
            return

        text=self.json_box.get(
            "1.0",
            "end"
        ).strip()

        data=json.loads(text)

        if not isinstance(data,dict):
            raise ValueError(
                "JSON must be an object."
            )

        self.config=data

        if self.autosave_var.get():
            self.active_config_path.write_text(
                text,
                encoding="utf-8"
            )
            self.write_env_settings()

    def load_config(self,path):
        path=Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file was not found:\n\n{path}"
            )

        data=json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data,dict):
            raise ValueError(
                f"{path.name} must contain a JSON object."
            )

        self.config=data
        self.active_config_path=path
        self.active_config_name=self.relative_config_path()

        self.json_box.delete(
            "1.0",
            "end"
        )

        self.json_box.insert(
            "1.0",
            json.dumps(
                self.config,
                indent=4,
                ensure_ascii=False
            )
        )

        if hasattr(self,"config_menu"):
            self.config_menu.set(
                self.relative_display_path(path)
            )

        self.console_log(
            f"Loaded {self.active_config_name}"
        )

    def json_changed(self,event=None):
        if not self.autosave_var.get():
            return

        if self.save_job:
            try:
                self.after_cancel(
                    self.save_job
                )
            except Exception:
                pass

        self.save_job=self.after(
            700,
            self.save_json
        )

    def save_json(self):
        self.save_job=None

        if not self.autosave_var.get():
            return

        try:
            text=self.json_box.get(
                "1.0",
                "end"
            ).strip()

            data=json.loads(text)

            if not isinstance(data,dict):
                raise ValueError(
                    "JSON must be an object."
                )

            self.config=data

            self.active_config_path.write_text(
                text,
                encoding="utf-8"
            )

            self.write_env_settings()

            self.console_log(
                f"{self.active_config_name} saved ✓"
            )

        except Exception as e:
            self.console_log(
                f"JSON error: {e}"
            )

    def sync_config(self):
        text=self.json_box.get(
            "1.0",
            "end"
        ).strip()

        data=json.loads(text)

        if not isinstance(data,dict):
            raise ValueError(
                "JSON must be an object."
            )

        self.config=data

        if self.autosave_var.get():
            self.active_config_path.write_text(
                text,
                encoding="utf-8"
            )
            self.write_env_settings()

    def unload_pipe(self):
        if self.pipe is not None:
            try:
                if (
                    DEVICE=="cuda"
                    and self.pipeline_dtype==torch.float16
                ):
                    self.pipe.to(
                        "cpu",
                        dtype=torch.float32
                    )
                else:
                    self.pipe.to("cpu")

            except Exception:
                pass

            del self.pipe
            self.pipe=None

        self.current_model_name=None
        self.current_model_id=None
        self.pipeline_dtype=None
        self.cleanup_gpu()

    def offload_diffusion_model(self):
        if self.pipe is None:
            return

        try:
            if (
                DEVICE=="cuda"
                and self.pipeline_dtype==torch.float16
            ):
                self.pipe.to(
                    "cpu",
                    dtype=torch.float32
                )
            else:
                self.pipe.to("cpu")
        except Exception:
            pass

        self.cleanup_gpu()

    def restore_diffusion_model(self):
        if self.pipe is None:
            return

        self.console_log(
            f"Returning {self.current_model_name} to {DEVICE.upper()}..."
        )

        if (
            DEVICE=="cuda"
            and self.pipeline_dtype==torch.float16
        ):
            self.pipe.to(
                DEVICE,
                dtype=torch.float16
            )
        else:
            self.pipe.to(DEVICE)

        self.cleanup_gpu()

    def model_changed(self,choice):
        if self.processing or self.model_loading:
            return

        if choice not in MODEL_OPTIONS:
            return

        if (
            self.current_model_name==choice
            and self.models_ready
        ):
            return

        self.models_ready=False

        self.generate_btn.configure(
            state="disabled"
        )

        self.model_menu.configure(
            state="disabled"
        )

        self.model_loading=True

        threading.Thread(
            target=self.load_models,
            args=(choice,),
            daemon=True
        ).start()

    def load_models(self,model_name):
        try:
            model_id=MODEL_OPTIONS.get(model_name,"")

            if model_name=="LazyMixRealAmateur v4.0 Inpainting":
                if not LAZYMIX_MODEL or not LAZYMIX_MODEL.exists():
                    raise FileNotFoundError(
                        'You didn\'t have LazyMix Model yet. Put it in the .env file or in the model folder as "lazymixRealAmateur_v40Inpainting.safetensors".'
                    )
                model_id=str(LAZYMIX_MODEL)

            if model_name=="DreamShaper 8 Inpainting":
                if not DREAMSHAPER_MODEL or not DREAMSHAPER_MODEL.exists():
                    raise FileNotFoundError(
                        'You didn\'t have DreamShaper Model yet. Put it in the .env file or in the model folder as "DreamShaper_8_INPAINTING.inpainting.safetensors".'
                    )
                model_id=str(DREAMSHAPER_MODEL)

            self.unload_pipe()

            dtype=(
                torch.float16
                if DEVICE=="cuda"
                else torch.float32
            )

            self.console_log(
                f"Loading {model_name}..."
            )

            try:
                pipe=StableDiffusionInpaintPipeline.from_single_file(
                    model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=True
                )
            except TypeError:
                pipe=StableDiffusionInpaintPipeline.from_single_file(
                    model_id,
                    torch_dtype=dtype,
                    safety_checker=None
                )

            pipe.scheduler=DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )

            pipe=pipe.to(DEVICE)
            pipe.enable_attention_slicing()

            if DEVICE=="cuda":
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass

            self.pipe=pipe
            self.current_model_name=model_name
            self.current_model_id=model_id
            self.pipeline_dtype=dtype

            self.cleanup_gpu()

            self.models_ready=True
            self.model_loading=False

            self.console_log(
                f"{model_name} ready • {self.active_config_name} • {DEVICE.upper()}"
            )

            self.after(
                0,
                lambda:self.model_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.generate_btn.configure(
                    state=(
                        "normal"
                        if self.original_image is not None
                        and not self.classification_loading
                        else "disabled"
                    )
                )
            )

        except Exception as e:
            self.models_ready=False
            self.model_loading=False
            self.segmentation_loading=False

            self.console_log(
                str(e)
            )

            self.cleanup_gpu()

            self.after(
                0,
                lambda:self.model_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda err=str(e):messagebox.showerror(
                    "Model Error",
                    err
                )
            )

    def set_image_class_from_result(self,classification):
        best_class=str(
            classification.get(
                "best_class",
                ""
            )
        ).strip()

        if not best_class:
            best_class="real life"

        self.image_class_var.set(
            best_class
        )

        self.image_class_menu.configure(
            state="normal"
        )

        best_probability=float(
            classification.get(
                "best_probability",
                0.0
            )
        )

        self.console_log(
            f"Image class: {best_class.upper()} ({best_probability*100:.1f}%)"
        )

    def set_gender_from_result(self,result):
        detected_gender=str(
            result[0]
        ).strip().lower()

        confidence=float(result[1])

        if detected_gender not in (
            "male",
            "female",
            "neutral"
        ):
            detected_gender="neutral"

        self.gender_var.set(
            detected_gender
        )

        self.gender_menu.configure(
            state="normal"
        )

        self.console_log(
            f"Gender: {detected_gender.upper()} ({confidence:.1f}%)"
        )

    def predict_gender_image(self,file_path):
        model=getattr(
            gender,
            "model",
            None
        )

        original_device=getattr(
            gender,
            "device",
            torch.device("cpu")
        )

        if model is None:
            raise RuntimeError(
                "gender.py model is not available."
            )

        try:
            model.to(original_device)
            return gender.predict_gender(
                file_path
            )
        finally:
            try:
                model.to("cpu")
            except Exception:
                pass

            self.cleanup_gpu()

    def classify_after_upload(self,file_path):
        try:
            self.console_log(
                "Classifying input image with imgclass..."
            )

            classification=self.classify_input_image(
                file_path
            )

            self.classification_result=classification

            self.after(
                0,
                lambda result=classification:self.set_image_class_from_result(result)
            )

            self.console_log(
                "Classifying input image with gender.py..."
            )

            gender_result=self.predict_gender_image(
                file_path
            )

            self.gender_result=gender_result

            self.after(
                0,
                lambda result=gender_result:self.set_gender_from_result(result)
            )

            self.console_log(
                "Image class and gender classification complete"
            )

        except Exception as e:
            self.classification_result=None
            self.gender_result=None

            self.after(
                0,
                lambda err=str(e):messagebox.showerror(
                    "Image Classification Error",
                    err
                )
            )

            self.after(
                0,
                lambda:self.image_class_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.gender_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.image_class_var.set(
                    "real life"
                )
            )

            self.after(
                0,
                lambda:self.gender_var.set(
                    "neutral"
                )
            )

            self.console_log(
                f"Image classification error: {e}"
            )

        finally:
            self.classification_loading=False

            self.after(
                0,
                lambda:self.generate_btn.configure(
                    state=(
                        "normal"
                        if self.models_ready
                        and self.original_image is not None
                        and not self.model_loading
                        and not self.segmentation_loading
                        else "disabled"
                    )
                )
            )

    def upload(self):
        if (
            self.processing
            or self.model_loading
            or self.segmentation_loading
            or self.classification_loading
        ):
            return

        path=filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                ("Images","*.png *.jpg *.jpeg *.webp *.bmp")
            ]
        )

        if not path:
            return

        try:
            image=ImageOps.exif_transpose(
                Image.open(path)
            ).convert("RGB")

            self.input_path=path
            self.original_image=image.copy()
            self.input_image=image.copy()
            self.output_image=None
            self.mask_image=None
            self.classification_result=None
            self.gender_result=None
            self.classification_loading=True

            self.image_class_var.set(
                "real life"
            )

            self.gender_var.set(
                "neutral"
            )

            self.image_class_menu.configure(
                state="disabled"
            )

            self.gender_menu.configure(
                state="disabled"
            )

            self.save_btn.configure(
                state="disabled"
            )

            self.generate_btn.configure(
                state="disabled"
            )

            self.show_input()
            self.show_output()

            self.console_log(
                f"Image loaded • {image.width}x{image.height} • Classifying..."
            )

            threading.Thread(
                target=self.classify_after_upload,
                args=(path,),
                daemon=True
            ).start()

        except Exception as e:
            messagebox.showerror(
                "Image Error",
                str(e)
            )

    def resize_image(self,image):
        w,h=image.size
        longest=max(w,h)

        if longest<=RESIZE_TARGET:
            return image

        scale=RESIZE_TARGET/longest

        nw=max(
            64,
            (int(w*scale)//8)*8
        )

        nh=max(
            64,
            (int(h*scale)//8)*8
        )

        self.console_log(
            f"Auto resize • {w}x{h} → {nw}x{nh}"
        )

        return image.resize(
            (nw,nh),
            Image.LANCZOS
        )

    def resize_output(self,image):
        w,h=image.size
        longest=max(w,h)

        if longest<=OUTPUT_TARGET:
            return image

        scale=OUTPUT_TARGET/longest

        nw=max(
            8,
            int(round(w*scale))
        )

        nh=max(
            8,
            int(round(h*scale))
        )

        return image.resize(
            (nw,nh),
            Image.LANCZOS
        )

    def load_esrgan(self):
        if self.esrgan:
            return

        if not RRDBNet or not RealESRGANer:
            raise RuntimeError(
                "Real-ESRGAN dependencies are not installed."
            )

        ESRGAN_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not ESRGAN_PATH.exists():
            self.console_log(
                "Downloading Real-ESRGAN model..."
            )

            urllib.request.urlretrieve(
                ESRGAN_URL,
                str(ESRGAN_PATH)
            )

        self.console_log(
            "Loading Real-ESRGAN..."
        )

        self.esrgan=RealESRGANer(
            scale=4,
            model_path=str(ESRGAN_PATH),
            model=RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            ),
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=DEVICE=="cuda",
            gpu_id=0 if DEVICE=="cuda" else None
        )

    def enhance(self,image,output=False):
        self.load_esrgan()

        self.console_log(
            "Enhancing output with Real-ESRGAN..."
            if output
            else
            "Enhancing original with Real-ESRGAN..."
        )

        bgr=cv2.cvtColor(
            np.asarray(image),
            cv2.COLOR_RGB2BGR
        )

        out,_=self.esrgan.enhance(
            bgr,
            outscale=4
        )

        self.cleanup_gpu()

        result=Image.fromarray(
            cv2.cvtColor(
                out,
                cv2.COLOR_BGR2RGB
            )
        ).convert("RGB")

        if output:
            result=self.resize_output(result)

            self.console_log(
                f"Output enhancement complete • {result.width}x{result.height}"
            )

        return result

    def apply_mask_adjustments(self,mask,thickness):
        array=np.asarray(
            mask,
            dtype=np.uint8
        )

        binary=(
            array>=128
        ).astype(np.uint8)

        thickness=float(thickness)

        if thickness<0:
            raise ValueError(
                "mask_outline_thickness cannot be negative."
            )

        if thickness>0:
            radius=int(
                round(thickness)
            )

            if radius>0:
                kernel_size=radius*2+1

                kernel=cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size,kernel_size)
                )

                binary=cv2.dilate(
                    binary,
                    kernel,
                    iterations=1
                )

        return Image.fromarray(
            (binary*255).astype(np.uint8),
            "L"
        )

    def ensure_segmentation(self):
        if (
            not hasattr(self,"segmentation")
            or self.segmentation is None
        ):
            self.segmentation=segdinosam2.SegDinoSAM2()

    def load_segmentation(self):
        self.ensure_segmentation()
        self.segmentation_loading=True

        try:
            self.console_log(
                "Loading segdinosam2 models..."
            )

            self.segmentation.load_dino()
            self.segmentation.load_sam2()

            self.console_log(
                "segdinosam2 ready"
            )

        finally:
            self.segmentation_loading=False

    def unload_segmentation(self):
        if (
            not hasattr(self,"segmentation")
            or self.segmentation is None
        ):
            return

        try:
            self.segmentation.unload_dino()
        except Exception:
            pass

        try:
            self.segmentation.unload_sam2()
        except Exception:
            pass

        gc.collect()

        if DEVICE=="cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def make_mask(self,image):
        segmentation_positive_prompt=str(
            self.config.get(
                "segmentation_positive_prompt",
                ""
            )
        )

        segmentation_negative_prompt=str(
            self.config.get(
                "segmentation_negative_prompt",
                ""
            )
        )

        thickness=float(
            self.config.get(
                "mask_outline_thickness",
                3.0
            )
        )

        blur=float(
            self.config.get(
                "mask_blur",
                4
            )
        )

        if thickness<0:
            raise ValueError(
                "mask_outline_thickness cannot be negative."
            )

        if blur<0:
            raise ValueError(
                "mask_blur cannot be negative."
            )

        self.load_segmentation()

        try:
            self.console_log(
                "Running segdinosam2 segmentation..."
            )

            result=self.segmentation.segment(
                image,
                segmentation_positive_prompt,
                segmentation_negative_prompt,
                thickness=0
            )

            masks=getattr(
                result,
                "masks",
                None
            )

            if masks is None and isinstance(result,dict):
                masks=result.get("masks")

            if masks is None:
                mask_image=getattr(
                    result,
                    "mask_image",
                    None
                )

                if (
                    mask_image is None
                    and isinstance(result,dict)
                ):
                    mask_image=result.get(
                        "mask_image"
                    )

                if mask_image is None:
                    raise RuntimeError(
                        "segdinosam2 returned no masks."
                    )

                raw_mask=Image.fromarray(
                    np.asarray(
                        mask_image,
                        dtype=np.uint8
                    ),
                    "L"
                )

                count=1

            else:
                masks=[
                    np.asarray(
                        mask,
                        dtype=bool
                    )
                    for mask in masks
                    if np.asarray(mask).any()
                ]

                if not masks:
                    raise RuntimeError(
                        "segdinosam2 returned no usable masks."
                    )

                combined=np.zeros(
                    image.size[::-1],
                    dtype=bool
                )

                for mask in masks:
                    if mask.shape!=combined.shape:
                        resized=Image.fromarray(
                            (mask.astype(np.uint8)*255),
                            "L"
                        ).resize(
                            image.size,
                            Image.Resampling.NEAREST
                        )

                        mask=np.asarray(
                            resized,
                            dtype=np.uint8
                        )>0

                    combined|=mask

                raw_mask=Image.fromarray(
                    (combined.astype(np.uint8)*255),
                    "L"
                )

                count=len(masks)

            if raw_mask.size!=image.size:
                raw_mask=raw_mask.resize(
                    image.size,
                    Image.Resampling.LANCZOS
                )

            mask=self.apply_mask_adjustments(
                raw_mask,
                thickness
            )

            if blur>0:
                mask=mask.filter(
                    ImageFilter.GaussianBlur(
                        radius=blur
                    )
                )

            mask_array=np.asarray(
                mask,
                dtype=np.uint8
            )

            self.console_log(
                f"Segmentation complete • masks={count} • coverage={mask_array.mean()/255*100:.1f}% • thickness={thickness:g} • blur={blur:g}"
            )

            return mask

        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def extract_original_face_hair(self,original_image):
        self.load_segmentation()

        try:
            self.console_log(
                "Segmenting original face and hair..."
            )

            result=self.segmentation.segment(
                original_image,
                FACE_RESTORE_POSITIVE_PROMPT,
                FACE_RESTORE_NEGATIVE_PROMPT,
                thickness=0
            )

            masks=getattr(
                result,
                "masks",
                None
            )

            if masks is None and isinstance(result,dict):
                masks=result.get("masks")

            if masks is None:
                mask_image=getattr(
                    result,
                    "mask_image",
                    None
                )

                if (
                    mask_image is None
                    and isinstance(result,dict)
                ):
                    mask_image=result.get(
                        "mask_image"
                    )

                if mask_image is None:
                    raise RuntimeError(
                        "segdinosam2 returned no face/hair mask."
                    )

                raw_mask=Image.fromarray(
                    np.asarray(
                        mask_image,
                        dtype=np.uint8
                    ),
                    "L"
                )

                count=1

            else:
                valid_masks=[]

                for mask in masks:
                    array=np.asarray(
                        mask
                    ).astype(bool)

                    if array.ndim>2:
                        array=np.squeeze(array)

                    if (
                        array.ndim!=2
                        or not np.any(array)
                    ):
                        continue

                    valid_masks.append(array)

                if not valid_masks:
                    raise RuntimeError(
                        "segdinosam2 returned no usable face/hair masks."
                    )

                combined=np.zeros(
                    original_image.size[::-1],
                    dtype=bool
                )

                for mask in valid_masks:
                    if mask.shape!=combined.shape:
                        resized=Image.fromarray(
                            (mask.astype(np.uint8)*255),
                            "L"
                        ).resize(
                            original_image.size,
                            Image.Resampling.NEAREST
                        )

                        mask=np.asarray(
                            resized,
                            dtype=np.uint8
                        )>0

                    combined|=mask

                raw_mask=Image.fromarray(
                    (combined.astype(np.uint8)*255),
                    "L"
                )

                count=len(valid_masks)

            if raw_mask.size!=original_image.size:
                raw_mask=raw_mask.resize(
                    original_image.size,
                    Image.Resampling.NEAREST
                )

            mask_array=np.asarray(
                raw_mask,
                dtype=np.uint8
            )

            area=np.count_nonzero(
                mask_array
            )

            if area<100:
                raise RuntimeError(
                    "No usable face/hair area was detected."
                )

            self.console_log(
                f"Original face/hair segmentation complete • masks={count} • coverage={mask_array.mean()/255*100:.1f}%"
            )

            return raw_mask

        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def paste_original_face_hair(self,original_image,output_image):
        original_image=original_image.convert(
            "RGB"
        )

        output_image=output_image.convert(
            "RGB"
        )

        face_hair_mask=self.extract_original_face_hair(
            original_image
        )

        if output_image.size!=original_image.size:
            original_for_output=original_image.resize(
                output_image.size,
                Image.Resampling.LANCZOS
            )

            face_hair_mask=face_hair_mask.resize(
                output_image.size,
                Image.Resampling.NEAREST
            )

        else:
            original_for_output=original_image

        mask_array=np.asarray(
            face_hair_mask,
            dtype=np.uint8
        )

        if np.count_nonzero(mask_array)<100:
            raise RuntimeError(
                "No usable original face/hair area was detected."
            )

        result=Image.composite(
            original_for_output,
            output_image,
            face_hair_mask
        ).convert("RGB")

        self.console_log(
            "Original face and hair pasted onto output"
        )

        return result

    def append_gender_prompts(
        self,
        positive_prompt,
        negative_prompt,
        selected_gender
    ):
        gender_positive={
            "male":"male character",
            "female":"female character",
            "neutral":""
        }

        gender_negative={
            "male":"",
            "female":"",
            "neutral":""
        }

        positive_append=gender_positive.get(
            selected_gender,
            ""
        )

        negative_append=gender_negative.get(
            selected_gender,
            ""
        )

        positive_prompt=positive_prompt.strip()
        negative_prompt=negative_prompt.strip()

        if positive_append:
            positive_prompt=(
                f"{positive_append}, {positive_prompt}"
                if positive_prompt
                else positive_append
            )

        if negative_append:
            negative_prompt=(
                f"{negative_append}, {negative_prompt}"
                if negative_prompt
                else negative_append
            )

        return positive_prompt,negative_prompt

    def append_classification_prompts(
        self,
        positive_prompt,
        negative_prompt,
        classification
    ):
        style_positive={
            "real life":"real",
            "anime":"anime",
            "3D":"3D",
            "cartoon":"cartoon"
        }

        style_negative={
            "real life":"anime, cartoon, 3D",
            "anime":"real life, 3D, cartoon",
            "3D":"real life, anime, cartoon",
            "cartoon":"real life, anime, 3D"
        }

        positive_append=style_positive.get(
            classification,
            classification
        )

        negative_append=style_negative.get(
            classification,
            ""
        )

        positive_prompt=positive_prompt.strip()
        negative_prompt=negative_prompt.strip()

        if positive_append:
            positive_prompt=(
                f"{positive_append}, {positive_prompt}"
                if positive_prompt
                else positive_append
            )

        if negative_append:
            negative_prompt=(
                f"{negative_append}, {negative_prompt}"
                if negative_prompt
                else negative_append
            )

        return positive_prompt,negative_prompt

    def generate(self):
        if self.processing:
            return

        if (
            self.model_loading
            or self.segmentation_loading
        ):
            messagebox.showwarning(
                "Models Loading",
                "Please wait until the models finish loading."
            )
            return

        if self.classification_loading:
            messagebox.showwarning(
                "Image Classification",
                "Please wait until the image class and gender are detected."
            )
            return

        if not self.models_ready:
            messagebox.showwarning(
                "Models Not Ready",
                "Please wait until the models finish loading."
            )
            return

        if self.original_image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )
            return

        try:
            self.sync_config()

        except Exception as e:
            messagebox.showerror(
                "Configuration Error",
                str(e)
            )
            return

        self.processing=True

        self.generate_btn.configure(
            state="disabled"
        )

        self.upload_btn.configure(
            state="disabled"
        )

        self.save_btn.configure(
            state="disabled"
        )

        self.model_menu.configure(
            state="disabled"
        )

        self.image_class_menu.configure(
            state="disabled"
        )

        self.gender_menu.configure(
            state="disabled"
        )

        self.start_time=time.time()

        self.console_log(
            "Starting generation..."
        )

        threading.Thread(
            target=self.worker,
            args=(
                self.original_image.copy(),
                self.current_model_name
            ),
            daemon=True
        ).start()

    def classify_input_image(self,file_path):
        return imgclass.classify_image(
            file_path
        )

    def composite_mask(self,base,generated,mask):
        if generated.size!=base.size:
            generated=generated.resize(
                base.size,
                Image.LANCZOS
            )

        if mask.size!=base.size:
            mask=mask.resize(
                base.size,
                Image.LANCZOS
            )

        return Image.composite(
            generated,
            base,
            mask
        ).convert("RGB")

    def worker(self,source,model_name):
        try:
            original_source=source.copy()

            if not self.input_path:
                raise RuntimeError(
                    "No input image path is available for image classification."
                )

            selected_class=str(
                self.image_class_var.get()
            ).strip()

            if not selected_class:
                selected_class="real life"

            selected_gender=str(
                self.gender_var.get()
            ).strip().lower()

            if selected_gender not in (
                "male",
                "female",
                "neutral"
            ):
                selected_gender="neutral"

            classification=self.classification_result or {}

            sorted_results=classification.get(
                "sorted_results",
                []
            )

            best_probability=float(
                classification.get(
                    "best_probability",
                    0.0
                )
            )

            detected_class=str(
                classification.get(
                    "best_class",
                    "unknown"
                )
            )

            gender_result=self.gender_result or (
                "neutral",
                0.0
            )

            detected_gender=str(
                gender_result[0]
            ).strip().lower()

            gender_confidence=float(
                gender_result[1]
            )

            self.console_log(
                f"Overall: image_class={selected_class.upper()} • gender={selected_gender.upper()}"
            )

            self.console_log(
                f"Detected image class: {detected_class.upper()} ({best_probability*100:.1f}%)"
            )

            self.console_log(
                f"Detected gender: {detected_gender.upper()} ({gender_confidence:.1f}%)"
            )

            steps=int(
                self.config.get(
                    "steps",
                    50
                )
            )

            cfg=float(
                self.config.get(
                    "cfg",
                    7.5
                )
            )

            strength=float(
                self.config.get(
                    "strength",
                    0.99
                )
            )

            seed=int(
                self.config.get(
                    "seed",
                    -1
                )
            )

            guidance_rescale=float(
                self.config.get(
                    "guidance_rescale",
                    0.0
                )
            )

            positive_prompt=str(
                self.config.get(
                    "positive_prompt",
                    ""
                )
            )

            negative_prompt=str(
                self.config.get(
                    "negative_prompt",
                    ""
                )
            )

            positive_prompt,negative_prompt=self.append_gender_prompts(
                positive_prompt,
                negative_prompt,
                selected_gender
            )

            positive_prompt,negative_prompt=self.append_classification_prompts(
                positive_prompt,
                negative_prompt,
                selected_class
            )

            self.console_log(
                f"Actual positive prompt: {positive_prompt}"
            )

            self.console_log(
                f"Actual negative prompt: {negative_prompt}"
            )

            self.console_log(
                f"Generating with selected class {selected_class.upper()} and gender {selected_gender.upper()}"
            )

            image=source

            if self.esrgan_input_var.get():
                image=self.enhance(
                    image,
                    output=False
                )

            if self.resize_var.get():
                image=self.resize_image(
                    image
                )

            self.input_image=image
            self.mask_image=None

            self.after(
                0,
                self.show_input
            )

            mask=self.make_mask(
                image
            )

            self.mask_image=mask

            self.after(
                0,
                self.show_input
            )

            if np.asarray(mask).max()<10:
                raise RuntimeError(
                    "No selected segmentation area was detected by segdinosam2."
                )

            w,h=self.generation_size(
                *image.size
            )

            init=image.resize(
                (w,h),
                Image.LANCZOS
            )

            mask=mask.resize(
                (w,h),
                Image.LANCZOS
            )

            if seed==-1:
                seed=torch.randint(
                    0,
                    2**32-1,
                    (1,)
                ).item()

            generator=torch.Generator(
                device=DEVICE
            ).manual_seed(
                seed
            )

            if self.pipe is None:
                raise RuntimeError(
                    "Selected diffusion model is not loaded."
                )

            def progress(
                pipe,
                step_index,
                timestep,
                callback_kwargs
            ):
                step=step_index+1
                elapsed=time.time()-self.start_time
                rate=elapsed/step
                remaining=(steps-step)*rate
                percent=int(
                    step/steps*100
                )

                self.console_log(
                    f"{percent:3d}% | {step}/{steps} | [{self.time_text(elapsed)}<{self.time_text(remaining)}] | {model_name}"
                )

                return callback_kwargs

            self.console_log(
                f"Generating {w}x{h} with {model_name}..."
            )

            self.pipe.scheduler=DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            result=self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=init,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength,
                generator=generator,
                width=w,
                height=h,
                callback_on_step_end=progress,
                guidance_rescale=guidance_rescale
            ).images[0]

            self.cleanup_gpu()

            if result.size!=image.size:
                result=result.resize(
                    image.size,
                    Image.LANCZOS
                )

            self.console_log(
                f"{model_name} generation complete"
            )

            self.offload_diffusion_model()

            try:
                result=self.paste_original_face_hair(
                    original_source,
                    result
                )
            finally:
                self.restore_diffusion_model()

            self.cleanup_gpu()

            if self.esrgan_output_var.get():
                result=self.enhance(
                    result,
                    output=True
                )

                if (
                    result.size[0]>OUTPUT_TARGET
                    or result.size[1]>OUTPUT_TARGET
                ):
                    result=self.resize_output(
                        result
                    )

                self.console_log(
                    f"Output Real-ESRGAN complete • {result.width}x{result.height}"
                )

            self.output_image=result

            self.cleanup_gpu()

            self.after(
                0,
                self.show_output
            )

            self.after(
                0,
                lambda:self.save_btn.configure(
                    state="normal"
                )
            )

            result_text=" | ".join(
                f"{name}={probability*100:.1f}%"
                for name,probability in sorted_results
            )

            self.console_log(
                f"Done • {result.width}x{result.height} • model={model_name} • seed={seed} • class={selected_class.upper()} • gender={selected_gender.upper()} • detected={detected_class.upper()} ({best_probability*100:.1f}%) • detected_gender={detected_gender.upper()} ({gender_confidence:.1f}%) • {result_text}"
            )

        except torch.cuda.OutOfMemoryError:
            self.cleanup_gpu()

            self.console_log(
                "Out of VRAM"
            )

            self.after(
                0,
                lambda:messagebox.showerror(
                    "Generation Error",
                    "Out of VRAM during generation."
                )
            )

        except Exception as e:
            self.cleanup_gpu()

            self.console_log(
                f"Generation error: {e}"
            )

            self.after(
                0,
                lambda err=str(e):messagebox.showerror(
                    "Generation Error",
                    err
                )
            )

        finally:
            self.cleanup_gpu()

            self.processing=False

            self.after(
                0,
                lambda:self.upload_btn.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.model_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.image_class_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.gender_menu.configure(
                    state="normal"
                )
            )

            self.after(
                0,
                lambda:self.generate_btn.configure(
                    state=(
                        "normal"
                        if self.models_ready
                        and self.original_image is not None
                        and not self.model_loading
                        and not self.segmentation_loading
                        and not self.classification_loading
                        else "disabled"
                    )
                )
            )

    def generation_size(self,w,h):
        scale=min(
            MAX_SIDE/w,
            MAX_SIDE/h,
            1.0
        )

        nw=max(
            64,
            int(w*scale)
        )

        nh=max(
            64,
            int(h*scale)
        )

        return (
            max(64,nw//8*8),
            max(64,nh//8*8)
        )

    def time_text(self,value):
        m,s=divmod(
            int(value),
            60
        )

        return f"{m:02d}:{s:02d}"

    def preview(self,image,canvas):
        if image is None:
            return None

        w=max(
            1,
            canvas.winfo_width()
        )

        h=max(
            1,
            canvas.winfo_height()
        )

        image=image.copy()

        image.thumbnail(
            (
                max(100,w-20),
                max(100,h-20)
            ),
            Image.LANCZOS
        )

        return image

    def overlay_mask(self,image):
        if (
            image is None
            or self.mask_image is None
        ):
            return image

        mask=self.mask_image

        if mask.size!=image.size:
            mask=mask.resize(
                image.size,
                Image.NEAREST
            )

        base=image.convert("RGBA")

        overlay=Image.new(
            "RGBA",
            image.size,
            (255,30,30,0)
        )

        alpha=Image.fromarray(
            np.where(
                np.asarray(mask)>0,
                110,
                0
            ).astype(np.uint8),
            "L"
        )

        overlay.putalpha(alpha)

        return Image.alpha_composite(
            base,
            overlay
        ).convert("RGB")

    def show_input(self):
        self.input_canvas.delete(
            "all"
        )

        if self.input_image is None:
            self.input_canvas.create_text(
                self.input_canvas.winfo_width()//2,
                self.input_canvas.winfo_height()//2,
                text="Upload an image",
                fill="#888888"
            )
            return

        image=(
            self.overlay_mask(
                self.input_image
            )
            if self.mask_var.get()
            else
            self.input_image
        )

        image=self.preview(
            image,
            self.input_canvas
        )

        if image is None:
            return

        self.input_photo=ImageTk.PhotoImage(
            image
        )

        self.input_canvas.create_image(
            self.input_canvas.winfo_width()//2,
            self.input_canvas.winfo_height()//2,
            image=self.input_photo
        )

    def show_output(self):
        self.output_canvas.delete(
            "all"
        )

        if self.output_image is None:
            self.output_canvas.create_text(
                self.output_canvas.winfo_width()//2,
                self.output_canvas.winfo_height()//2,
                text="Output will appear here",
                fill="#888888"
            )
            return

        image=self.preview(
            self.output_image,
            self.output_canvas
        )

        if image is None:
            return

        self.output_photo=ImageTk.PhotoImage(
            image
        )

        self.output_canvas.create_image(
            self.output_canvas.winfo_width()//2,
            self.output_canvas.winfo_height()//2,
            image=self.output_photo
        )

    def save(self):
        if self.output_image is None:
            messagebox.showwarning(
                "No Output",
                "There is no generated image to save."
            )
            return

        name=(
            "output.png"
            if not self.input_path
            else
            Path(self.input_path).stem+"_output.png"
        )

        path=filedialog.asksaveasfilename(
            title="Save Image As",
            initialfile=name,
            defaultextension=".png",
            filetypes=[
                ("PNG","*.png"),
                ("JPEG","*.jpg *.jpeg"),
                ("WebP","*.webp")
            ]
        )

        if not path:
            return

        try:
            ext=Path(path).suffix.lower()

            if ext in (
                ".jpg",
                ".jpeg"
            ):
                self.output_image.convert(
                    "RGB"
                ).save(
                    path,
                    quality=95
                )

            elif ext==".webp":
                self.output_image.save(
                    path,
                    quality=95
                )

            else:
                self.output_image.save(
                    path
                )

            self.console_log(
                f"Saved: {os.path.basename(path)}"
            )

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                str(e)
            )

    def destroy(self):
        try:
            sys.stdout=sys.__stdout__
            sys.stderr=sys.__stderr__
        except Exception:
            pass

        try:
            self.unload_pipe()
        except Exception:
            pass

        try:
            self.unload_segmentation()
        except Exception:
            pass

        try:
            gender_model=getattr(
                gender,
                "model",
                None
            )

            if gender_model is not None:
                gender_model.to("cpu")

        except Exception:
            pass

        self.esrgan=None
        self.cleanup_gpu()
        super().destroy()

if __name__=="__main__":
    app=App()
    app.mainloop()