import os, sys, json, time, threading, gc
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np, torch, cv2
import torchvision.transforms.functional as TF
sys.modules.setdefault('torchvision.transforms.functional_tensor', TF)
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import modules.segdinosam2 as segdinosam2
import modules.imgclass as imgclass
import modules.json_textbox as json_textbox
REAL = imgclass.REAL
ANIME = imgclass.ANIME
THREE_D = imgclass.THREE_D
CARTOON = imgclass.CARTOON
import modules.gender as gender
import model.upscale_img as upscale_img

try:
    gender.model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme(str(BASE_DIR / 'themes' / 'custom.json'))
MODEL_DIR = BASE_DIR / 'model'
DEFAULT_JSON = BASE_DIR / 'data/diffusion_config/default.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_env_file():
    env_path = BASE_DIR / '.env'
    values = {}
    if not env_path.exists():
        return values
    try:
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        return {}
    return values

def discover_models():
    env_value = read_env_file().get('SD_INPAINT_MODEL', '').strip()
    paths = []
    if env_value:
        try:
            parsed = json.loads(env_value)
            if isinstance(parsed, list):
                paths.extend(parsed)
            elif isinstance(parsed, str):
                paths.append(parsed)
        except json.JSONDecodeError:
            paths.append(env_value)
    if MODEL_DIR.exists():
        paths.extend(str(path) for path in sorted(MODEL_DIR.glob('*.safetensors'), key=lambda item: item.name.lower()))
    models = {}
    seen = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(str(raw_path).strip())
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve(strict=False)
        key = str(path).lower()
        if key in seen or not path.exists() or path.suffix.lower() != '.safetensors':
            continue
        seen.add(key)
        models[path.stem] = str(path)
    return models

MODEL_OPTIONS = {}
DEFAULT_MODEL = ''
MAX_SIDE = 768
MIN_RESIZE_SIDE = 512
OUTPUT_TARGET = 1080
IMAGE_CLASSES = [REAL, ANIME, THREE_D, CARTOON]
FACE_RESTORE_POSITIVE_PROMPT = 'face, hair'
FACE_RESTORE_NEGATIVE_PROMPT = 'neck'

class ConsoleRedirect:

    def __init__(self, app):
        self.app = app

    def write(self, text):
        if not text:
            return
        if '\r' in text:
            parts = text.split('\r')
            if len(parts) > 1:
                text = parts[-1]
            self.app.console_log(text, live=True)
        else:
            self.app.console_log(text)

    def flush(self):
        pass

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('DiffuVision AI - Inpainting with Stable Diffusion')
        self.iconbitmap('favicon.ico')
        self.geometry('1600x1150')
        self.minsize(1100, 900)
        self.pipe = None
        self.current_model_name = None
        self.current_model_id = None
        self.pipeline_dtype = None
        self.segmentation = None
        self.original_image = None
        self.input_image = None
        self.sd_input_image = None
        self.output_image = None
        self.input_path = None
        self.mask_image = None
        self.classification_result = None
        self.gender_result = None
        self.classification_loading = False
        self.processing = False
        self.models_ready = False
        self.model_loading = False
        self.segmentation_loading = False
        self.start_time = 0
        self.save_job = None
        self.input_photo = None
        self.output_photo = None
        self.crop_box = None
        self.crop_box_start = None
        self.input_display_info = None
        self.cropping = False
        self.active_crop_handle = None
        self.cropped_original_image = None
        self.resize_var = ctk.BooleanVar(value=True)
        self.esrgan_input_var = ctk.BooleanVar(value=False)
        self.esrgan_output_var = ctk.BooleanVar(value=False)
        self.face_restore_var = ctk.BooleanVar(value=False)
        self.mask_var = ctk.BooleanVar(value=True)
        self.autosave_var = ctk.BooleanVar(value=True)
        self.model_var = ctk.StringVar(value='')
        self.image_class_var = ctk.StringVar(value=REAL)
        self.gender_var = ctk.StringVar(value='neutral')
        self.config_files = []
        self.active_config_name = 'data/diffusion_config/default.json'
        self.active_config_path = DEFAULT_JSON
        self.config = {}
        self.env_path = BASE_DIR / '.env'
        self.output_showing_original = False
        self.load_env_settings()
        self.model_var.set(DEFAULT_MODEL)
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.ui()
        self.stdout_redirect = ConsoleRedirect(self)
        self.stderr_redirect = ConsoleRedirect(self)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self.after(100, self.maximize)
        self.refresh_config_files()
        self.load_startup_config()
        if self.model_var.get():
            threading.Thread(target=self.load_models, args=(self.model_var.get(),), daemon=True).start()

    def maximize(self):
        try:
            self.state('zoomed')
        except Exception:
            pass

    def ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.left_frame = ctk.CTkScrollableFrame(self)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
        self.left_frame.grid_columnconfigure(0, weight=0)
        self.left_frame.grid_columnconfigure(1, weight=1)
        self.input_label = ctk.CTkLabel(self.left_frame, text='INPUT IMAGE')
        self.input_label.grid(row=0, column=0, sticky='w', padx=4, pady=(0, 4))
        self.mask_button_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.mask_button_row.grid(row=0, column=1, sticky='e', padx=2, pady=(0, 4))
        self.mask_btn = ctk.CTkButton(self.mask_button_row, text='HIDE MASK', command=self.toggle_mask, width=120, height=34)
        self.mask_btn.grid(row=0, column=0, sticky='e', padx=(0, 3))
        self.reload_mask_btn = ctk.CTkButton(self.mask_button_row, text='RELOAD MASK', command=self.reload_mask, width=120, height=34, state='disabled')
        self.reload_mask_btn.grid(row=0, column=1, sticky='e', padx=(3, 0))
        self.input_image_container = ctk.CTkFrame(self.left_frame, fg_color='#030303', corner_radius=0, height=600)
        self.input_image_container.grid(row=1, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.input_image_container.grid_propagate(False)
        self.input_canvas = ctk.CTkCanvas(self.input_image_container, bg='#030303', highlightthickness=0)
        self.input_canvas.pack(fill='both', expand=True)
        self.reset_crop_btn = ctk.CTkButton(self.input_image_container, text='🖾', command=self.reset_crop, width=34, height=34, corner_radius=6, fg_color='#21262D', hover_color='#30363D', font=('Segoe UI Symbol', 17))
        self.reset_crop_btn.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.upload_btn = ctk.CTkButton(self.left_frame, text='UPLOAD IMAGE', command=self.upload, height=40)
        self.upload_btn.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.preprocessing_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.preprocessing_row.grid(row=3, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.preprocessing_row.grid_columnconfigure(0, weight=1)
        self.preprocessing_row.grid_columnconfigure(1, weight=1)
        self.resize_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Auto resize original image for recommended SD inpainting', variable=self.resize_var)
        self.resize_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.esrgan_input_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Enhance original image', variable=self.esrgan_input_var)
        self.esrgan_input_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        ctk.CTkLabel(self.left_frame, text='MODEL (SD INPAINTING):', anchor='w', width=100).grid(row=4, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.model_menu = ctk.CTkOptionMenu(self.left_frame, variable=self.model_var, values=list(MODEL_OPTIONS.keys()), command=self.model_changed)
        self.model_menu.grid(row=4, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.class_gender_row.grid(row=5, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row.grid_columnconfigure(0, weight=0)
        self.class_gender_row.grid_columnconfigure(1, weight=1)
        self.class_gender_row.grid_columnconfigure(2, weight=0)
        self.class_gender_row.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(self.class_gender_row, text='IMAGE CLASS:', anchor='w', width=100).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.image_class_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.image_class_var, values=IMAGE_CLASSES)
        self.image_class_menu.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.class_gender_row, text='GENDER:', anchor='w', width=75).grid(row=0, column=2, sticky='w', padx=(2, 8))
        self.gender_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.gender_var, values=['male', 'female', 'neutral'])
        self.gender_menu.grid(row=0, column=3, sticky='ew', padx=(0, 2))
        ctk.CTkLabel(self.left_frame, text='JSON CONFIG:', anchor='w', width=100).grid(row=6, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.config_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.config_row.grid(row=6, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.config_row.grid_columnconfigure(0, weight=1)
        self.config_row.grid_columnconfigure(1, weight=0)
        self.config_menu = ctk.CTkOptionMenu(self.config_row, values=[], command=self.config_changed)
        self.config_menu.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.autosave_btn = ctk.CTkButton(self.config_row, text='AUTOSAVE: ON', command=self.toggle_autosave, height=38, width=105)
        self.autosave_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.json_box = json_textbox.JSONTextBox(self.left_frame, height=220, font_size=12, fg_color='#000000')
        self.json_box.grid(row=7, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.json_box.set_change_callback(self.json_changed)
        self.output_options_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.output_options_row.grid(row=8, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 0))
        self.output_options_row.grid_columnconfigure(0, weight=1)
        self.output_options_row.grid_columnconfigure(1, weight=1)
        self.esrgan_output_cb = ctk.CTkCheckBox(self.output_options_row, text='Enhance output image', variable=self.esrgan_output_var)
        self.esrgan_output_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.face_restore_cb = ctk.CTkCheckBox(self.output_options_row, text='Face restore on output', variable=self.face_restore_var)
        self.face_restore_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        self.generate_btn = ctk.CTkButton(self.left_frame, text='GENERATE', command=self.generate, state='disabled', height=42)
        self.generate_btn.grid(row=9, column=0, columnspan=2, sticky='ew', padx=2, pady=(12, 8))
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.output_container = ctk.CTkFrame(self.right_frame, fg_color='#000000', corner_radius=6)
        self.output_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=(10, 6))
        self.output_container.grid_rowconfigure(0, weight=1)
        self.output_container.grid_columnconfigure(0, weight=1)
        self.output_canvas = ctk.CTkCanvas(self.output_container, bg='#000000', highlightthickness=0)
        self.output_canvas.grid(row=0, column=0, sticky='nsew')
        self.switch_image_btn = ctk.CTkButton(self.output_container, text='⇄', command=self.switch_output_image, width=34, height=34, corner_radius=6, font=('Segoe UI Symbol', 18), fg_color='#21262D', hover_color='#30363D')
        self.switch_image_btn.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.console = ctk.CTkTextbox(self.right_frame, height=260, wrap='none', font=('Consolas', 12), fg_color='#000000', text_color='#D0D0D0')
        self.console.grid(row=1, column=0, sticky='ew', padx=10, pady=6)
        self.console.configure(state='disabled')
        self.save_clear_row = ctk.CTkFrame(self.right_frame, fg_color='transparent')
        self.save_clear_row.grid(row=2, column=0, sticky='ew', padx=10, pady=(6, 10))
        self.save_clear_row.grid_columnconfigure(0, weight=1)
        self.save_clear_row.grid_columnconfigure(1, weight=0)
        self.save_btn = ctk.CTkButton(self.save_clear_row, text='SAVE IMAGE AS', command=self.save, state='disabled', height=40)
        self.save_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.clear_btn = ctk.CTkButton(self.save_clear_row, text='CLEAR', command=self.clear_console, height=40, width=100)
        self.clear_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.input_canvas.bind('<Configure>', lambda e: self.show_input())
        self.input_canvas.bind('<ButtonPress-1>', self.start_crop)
        self.input_canvas.bind('<B1-Motion>', self.update_crop_selection)
        self.input_canvas.bind('<ButtonRelease-1>', self.finish_crop)
        self.output_canvas.bind('<Configure>', lambda e: self.show_output())
        self.update_autosave_button()
        self.show_input()
        self.show_output()

    def toggle_mask(self):
        self.mask_var.set(not self.mask_var.get())
        self.mask_btn.configure(text='HIDE MASK' if self.mask_var.get() else 'SHOW MASK')
        self.show_input()

    def update_generate_state(self):
        state = 'normal' if self.models_ready and self.original_image is not None and self.mask_image is not None and self.sd_input_image is not None and not self.model_loading and not self.segmentation_loading and not self.classification_loading and not self.processing else 'disabled'
        self.generate_btn.configure(state=state)

    def load_env_settings(self):
        values = read_env_file()
        global MODEL_OPTIONS, DEFAULT_MODEL
        MODEL_OPTIONS = discover_models()
        DEFAULT_MODEL = next(iter(MODEL_OPTIONS), '')
        config_value = values.get('JSON_config', '').replace('\\', '/')
        autosave_value = values.get('JSON_autosave', None)
        if config_value:
            candidate = Path(config_value)
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate
            self.startup_config_path = candidate
        else:
            self.startup_config_path = DEFAULT_JSON
        if autosave_value is None:
            self.autosave_var.set(True)
        else:
            self.autosave_var.set(autosave_value.strip().lower() in ('1', 'true', 'yes', 'on'))

    def write_env_settings(self):
        values = {}
        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding='utf-8').splitlines():
                    line = line.strip()
                    if not line or line.startswith('#') or ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
            except Exception:
                pass
        values['JSON_config'] = self.relative_config_path()
        values['JSON_autosave'] = 'True' if self.autosave_var.get() else 'False'
        lines = []
        written = set()
        if self.env_path.exists():
            try:
                for line in self.env_path.read_text(encoding='utf-8').splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#') or ':' not in stripped:
                        lines.append(line)
                        continue
                    key = stripped.split(':', 1)[0].strip()
                    if key == 'JSON_config':
                        lines.append(f'JSON_config: "{values["JSON_config"]}"')
                        written.add(key)
                    elif key == 'JSON_autosave':
                        lines.append(f'JSON_autosave: {values["JSON_autosave"]}')
                        written.add(key)
                    else:
                        lines.append(line)
            except Exception:
                lines = []
        if 'JSON_config' not in written:
            lines.append(f'JSON_config: "{values["JSON_config"]}"')
        if 'JSON_autosave' not in written:
            lines.append(f'JSON_autosave: {values["JSON_autosave"]}')
        self.env_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def relative_config_path(self):
        try:
            return self.active_config_path.relative_to(BASE_DIR).as_posix()
        except Exception:
            return str(self.active_config_path).replace('\\', '/')

    def relative_display_path(self, path):
        try:
            return path.relative_to(BASE_DIR).as_posix()
        except Exception:
            return str(path).replace('\\', '/')

    def refresh_config_files(self):
        data_dir = BASE_DIR / 'data' / 'diffusion_config'
        data_dir.mkdir(parents=True, exist_ok=True)
        self.config_files = sorted([p for p in data_dir.glob('*.json') if p.is_file()], key=lambda p: p.name.lower())
        values = [self.relative_display_path(p) for p in self.config_files]
        self.config_menu.configure(values=values if values else ['data/diffusion_config/default.json'])

    def load_startup_config(self):
        self.refresh_config_files()
        candidates = self.config_files.copy()
        target = self.startup_config_path
        if not target.exists() or target.suffix.lower() != '.json':
            target = DEFAULT_JSON if DEFAULT_JSON.exists() else candidates[0] if candidates else target
        if target not in candidates and target.exists():
            candidates.append(target)
            candidates = sorted(candidates, key=lambda p: p.name.lower())
            self.config_files = candidates
            self.config_menu.configure(values=[self.relative_display_path(p) for p in candidates])
        if target.exists():
            self.load_config(target)
            self.config_menu.set(self.relative_display_path(target))
        self.update_autosave_button()
        self.write_env_settings()

    def config_changed(self, choice):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        target = BASE_DIR / choice
        if not target.exists():
            return
        try:
            self.load_config(target)
            self.write_env_settings()
        except Exception as e:
            messagebox.showerror('Configuration Error', str(e))

    def toggle_autosave(self):
        if self.processing:
            return
        self.autosave_var.set(not self.autosave_var.get())
        self.update_autosave_button()
        self.write_env_settings()
        if self.autosave_var.get():
            self.save_json()

    def update_autosave_button(self):
        if not hasattr(self, 'autosave_btn'):
            return
        if self.autosave_var.get():
            self.autosave_btn.configure(text='AUTOSAVE: ON', fg_color='#1f8f3a', hover_color='#176b2c')
        else:
            self.autosave_btn.configure(text='AUTOSAVE: OFF', fg_color='#666666', hover_color='#555555')

    def clear_console(self):
        try:
            self.console.configure(state='normal')
            self.console.delete('1.0', 'end')
            self.console.configure(state='disabled')
        except Exception:
            pass

    def console_log(self, text, color=None, live=False):
        if text is None:
            return
        text = str(text)
        text = text.replace('\x1b[K', '')
        text = text.replace('\x1b[2K', '')
        text = text.strip('\n')
        if not text:
            return

        def update():
            try:
                self.console.configure(state='normal')
                if live:
                    line_start = self.console.index('end-1c linestart')
                    self.console.delete(line_start, 'end-1c')
                    self.console.insert('end', text)
                else:
                    self.console.insert('end', text)
                    if not text.endswith('\n'):
                        self.console.insert('end', '\n')
                self.console.see('end')
                self.console.configure(state='disabled')
            except Exception:
                pass

        try:
            self.after(0, update)
        except Exception:
            pass

    def cleanup_gpu(self):
        try:
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()
        if DEVICE == 'cuda':
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
        text = self.json_box.get('1.0', 'end').strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        self.config = data
        if self.autosave_var.get():
            self.active_config_path.write_text(text, encoding='utf-8')
            self.write_env_settings()

    def load_config(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Configuration file was not found:\n\n{path}')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'{path.name} must contain a JSON object.')
        self.config = data
        self.active_config_path = path
        self.active_config_name = self.relative_config_path()
        self.json_box.delete('1.0', 'end')
        self.json_box.insert('1.0', json.dumps(self.config, indent=4, ensure_ascii=False))
        if hasattr(self, 'config_menu'):
            self.config_menu.set(self.relative_display_path(path))
        self.console_log(f'Loaded {self.active_config_name}')

    def json_changed(self, event=None):
        if not self.autosave_var.get():
            return
        if self.save_job:
            try:
                self.after_cancel(self.save_job)
            except Exception:
                pass
        self.save_job = self.after(700, self.save_json)

    def save_json(self):
        self.save_job = None
        if not self.autosave_var.get():
            return
        try:
            text = self.json_box.get('1.0', 'end').strip()
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError('JSON must be an object.')
            self.config = data
            self.active_config_path.write_text(text, encoding='utf-8')
            self.write_env_settings()
            self.console_log(f'{self.active_config_name} saved ✓')
        except Exception as e:
            self.console_log(f'JSON error: {e}')

    def sync_config(self):
        text = self.json_box.get('1.0', 'end').strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('JSON must be an object.')
        self.config = data
        if self.autosave_var.get():
            self.active_config_path.write_text(text, encoding='utf-8')
            self.write_env_settings()

    def unload_pipe(self):
        if self.pipe is not None:
            try:
                if DEVICE == 'cuda' and self.pipeline_dtype == torch.float16:
                    self.pipe.to('cpu', dtype=torch.float32)
                else:
                    self.pipe.to('cpu')
            except Exception:
                pass
            del self.pipe
            self.pipe = None
        self.current_model_name = None
        self.current_model_id = None
        self.pipeline_dtype = None
        self.cleanup_gpu()

    def offload_diffusion_model(self):
        if self.pipe is None:
            return
        try:
            if DEVICE == 'cuda' and self.pipeline_dtype == torch.float16:
                self.pipe.to('cpu', dtype=torch.float32)
            else:
                self.pipe.to('cpu')
        except Exception:
            pass
        self.cleanup_gpu()

    def restore_diffusion_model(self):
        if self.pipe is None:
            return
        self.console_log(f'Returning {self.current_model_name} to {DEVICE.upper()}...')
        if DEVICE == 'cuda' and self.pipeline_dtype == torch.float16:
            self.pipe.to(DEVICE, dtype=torch.float16)
        else:
            self.pipe.to(DEVICE)
        self.cleanup_gpu()

    def model_changed(self, choice):
        if self.processing or self.model_loading:
            return
        if choice not in MODEL_OPTIONS:
            return
        if self.current_model_name == choice and self.models_ready:
            return
        self.models_ready = False
        self.generate_btn.configure(state='disabled')
        self.model_menu.configure(state='disabled')
        self.model_loading = True
        threading.Thread(target=self.load_models, args=(choice,), daemon=True).start()

    def load_models(self, model_name):
        try:
            model_id = MODEL_OPTIONS.get(model_name, '')
            if not model_id or not Path(model_id).exists():
                raise FileNotFoundError(f'Model "{model_name}" was not found. Add a .safetensors file to the model folder or list it in "SD_INPAINT_MODEL" in .env.')
            self.unload_pipe()
            dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
            self.console_log(f'Loading {model_name}...')
            try:
                pipe = StableDiffusionInpaintPipeline.from_single_file(model_id, torch_dtype=dtype, safety_checker=None, local_files_only=True)
            except TypeError:
                pipe = StableDiffusionInpaintPipeline.from_single_file(model_id, torch_dtype=dtype, safety_checker=None)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(DEVICE)
            pipe.enable_attention_slicing()
            if DEVICE == 'cuda':
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            self.pipe = pipe
            self.current_model_name = model_name
            self.current_model_id = model_id
            self.pipeline_dtype = dtype
            self.cleanup_gpu()
            self.models_ready = True
            self.model_loading = False
            self.console_log(f'{model_name} ready • {self.active_config_name} • {DEVICE.upper()}')
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, self.update_generate_state)
        except Exception as e:
            self.models_ready = False
            self.model_loading = False
            self.segmentation_loading = False
            self.console_log(str(e))
            self.cleanup_gpu()
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda err=str(e): messagebox.showerror('Model Error', err))

    def set_image_class_from_result(self, classification):
        best_class = str(classification.get('best_class', '')).strip()
        if not best_class:
            best_class = REAL
        self.image_class_var.set(best_class)
        self.image_class_menu.configure(state='normal')
        best_probability = float(classification.get('best_probability', 0.0))
        self.console_log(f'Image class: {best_class.upper()} ({best_probability * 100:.1f}%)')

    def set_gender_from_result(self, result):
        detected_gender = str(result[0]).strip().lower()
        confidence = float(result[1])
        if detected_gender not in ('male', 'female', 'neutral'):
            detected_gender = 'neutral'
        self.gender_var.set(detected_gender)
        self.gender_menu.configure(state='normal')
        self.console_log(f'Gender: {detected_gender.upper()} ({confidence:.1f}%)')

    def predict_gender_image(self, file_path):
        model = getattr(gender, 'model', None)
        original_device = getattr(gender, 'device', torch.device('cpu'))
        if model is None:
            raise RuntimeError('gender.py model is not available.')
        try:
            model.to(original_device)
            return gender.predict_gender(file_path)
        finally:
            try:
                model.to('cpu')
            except Exception:
                pass
            self.cleanup_gpu()

    def prepare_uploaded_image(self, path):
        image = ImageOps.exif_transpose(Image.open(path))
        has_alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        if has_alpha:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert('RGB')
            self.console_log(f'Transparent background detected • filled with white • {image.width}x{image.height}')
        else:
            image = image.convert('RGB')
        return image

    def classify_after_upload(self, file_path):
        try:
            self.console_log('Classifying input image with imgclass...')
            classification = self.classify_input_image(file_path)
            self.classification_result = classification
            self.after(0, lambda result=classification: self.set_image_class_from_result(result))
            self.console_log('Classifying input image with gender.py...')
            gender_result = self.predict_gender_image(file_path)
            self.gender_result = gender_result
            self.after(0, lambda result=gender_result: self.set_gender_from_result(result))
            self.console_log('Image class and gender classification complete')
            self.console_log('Automatic segdinosam2 disabled after upload')
            self.console_log('Adjust crop and click RELOAD MASK when ready')
        except Exception as e:
            self.classification_result = None
            self.gender_result = None
            self.after(0, lambda err=str(e): messagebox.showerror('Image Classification Error', err))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_var.set(REAL))
            self.after(0, lambda: self.gender_var.set('neutral'))
            self.console_log(f'Image classification error: {e}')
        finally:
            self.classification_loading = False
            self.after(0, self.update_generate_state)

    def upload(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        path = filedialog.askopenfilename(title='Select input image', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.avif')])
        if not path:
            return
        try:
            image = self.prepare_uploaded_image(path)
            self.input_path = path
            self.original_image = image.copy()
            self.input_image = image.copy()
            self.sd_input_image = None
            self.cropped_original_image = self.get_cropped_image(image)
            self.output_image = None
            self.output_showing_original = False
            self.mask_image = None
            self.classification_result = None
            self.gender_result = None
            self.classification_loading = True
            self.image_class_var.set(REAL)
            self.gender_var.set('neutral')
            self.image_class_menu.configure(state='disabled')
            self.gender_menu.configure(state='disabled')
            self.save_btn.configure(state='disabled')
            self.generate_btn.configure(state='disabled')
            self.reload_mask_btn.configure(state='normal')
            self.reset_crop_btn.configure(state='normal' if self.crop_box is not None else 'disabled')
            self.show_input()
            self.show_output()
            self.console_log(f'Image loaded • white background applied when needed • {image.width}x{image.height} • Classifying...')
            threading.Thread(target=self.classify_after_upload, args=(path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror('Image Error', str(e))

    def resize_image(self, image):
        image = image.convert('RGB')
        w, h = image.size
        short_side = min(w, h)
        if short_side >= MIN_RESIZE_SIDE:
            return image
        scale = MIN_RESIZE_SIDE / short_side
        nw = max(MIN_RESIZE_SIDE, int(round(w * scale)))
        nh = max(MIN_RESIZE_SIDE, int(round(h * scale)))
        nw = max(MIN_RESIZE_SIDE, (nw // 8) * 8)
        nh = max(MIN_RESIZE_SIDE, (nh // 8) * 8)
        self.console_log(f'Auto resize • {w}x{h} → {nw}x{nh}')
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def resize_output(self, image):
        w, h = image.size
        longest = max(w, h)
        if longest <= OUTPUT_TARGET:
            return image
        scale = OUTPUT_TARGET / longest
        nw = max(8, int(round(w * scale)))
        nh = max(8, int(round(h * scale)))
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def apply_mask_adjustments(self, mask, thickness):
        array = np.asarray(mask, dtype=np.uint8)
        binary = (array >= 128).astype(np.uint8)
        thickness = float(thickness)
        if thickness < 0:
            raise ValueError('mask_outline_thickness cannot be negative.')
        if thickness > 0:
            radius = int(round(thickness))
            if radius > 0:
                kernel_size = radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                binary = cv2.dilate(binary, kernel, iterations=1)
        return Image.fromarray((binary * 255).astype(np.uint8), 'L')

    def ensure_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            self.segmentation = segdinosam2.SegDinoSAM2()

    def load_segmentation(self):
        self.ensure_segmentation()
        self.console_log('Loading segdinosam2 models...')
        self.segmentation.load_dino()
        self.segmentation.load_sam2()
        self.console_log('segdinosam2 ready')

    def unload_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
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
        if DEVICE == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def regenerate_mask(self, image):
        if image is None or self.processing or self.model_loading or self.classification_loading:
            return
        self.segmentation_loading = True
        self.reload_mask_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.after(0, self.show_output)
        try:
            self.console_log(f'Generating mask with segdinosam2 from {image.width}x{image.height} input...')
            mask = self.make_mask(image)
            self.mask_image = mask
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            self.console_log('Mask ready')
        except Exception as e:
            self.mask_image = None
            self.console_log(f'Mask error: {e}')
            self.after(0, lambda err=str(e): messagebox.showerror('Mask Error', err))
        finally:
            self.segmentation_loading = False
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing else 'disabled'))
            self.after(0, lambda: self.reset_crop_btn.configure(state='normal' if self.original_image is not None and self.crop_box is not None else 'disabled'))
            self.after(0, self.update_generate_state)

    def reload_mask(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        try:
            self.sync_config()
        except Exception as e:
            messagebox.showerror('Configuration Error', str(e))
            return
        self.mask_image = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False
        self.generate_btn.configure(state='disabled')
        self.reload_mask_btn.configure(state='disabled')
        self.show_input()
        self.show_output()
        self.console_log('RELOAD MASK • current JSON + current crop')
        self.console_log('Input preprocessing started...')
        threading.Thread(target=self.reload_mask_worker, daemon=True).start()

    def reload_mask_worker(self):
        try:
            processed = self.preprocess_uploaded_image(self.original_image)
            self.after(0, self.show_input)
            self.console_log(f'Starting segdinosam2 mask generation from {processed.width}x{processed.height}...')
            self.regenerate_mask(processed.copy())
        except Exception as e:
            self.console_log(f'RELOAD MASK error: {e}')
            self.after(0, lambda err=str(e): messagebox.showerror('Mask Reload Error', err))
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None else 'disabled'))

    def preprocess_uploaded_image(self, image):
        base = image.convert('RGB').copy()
        if self.resize_var.get():
            base = self.resize_image(base)
        self.input_image = base.copy()
        self.console_log(f'Pre-crop processing image • {base.width}x{base.height}')
        cropped = self.get_cropped_image(base)
        self.cropped_original_image = self.get_cropped_image(image)
        if self.crop_box is not None:
            self.console_log(f'Crop applied for preprocessing • {cropped.width}x{cropped.height}')
        else:
            self.console_log(f'No crop applied • {cropped.width}x{cropped.height}')
        processed = cropped.copy()
        if self.esrgan_input_var.get():
            self.console_log('Enhance original image started...')
            processed = upscale_img.enhance(processed, output=False, logger=self.console_log, output_target=OUTPUT_TARGET)
            self.console_log(f'Enhance original image complete • {processed.width}x{processed.height}')
        if self.resize_var.get():
            processed = self.resize_image(processed)
        self.sd_input_image = processed.copy()
        self.console_log(f'Preprocessed generation input • {processed.width}x{processed.height}')
        return processed

    def make_mask(self, image):
        segmentation_positive_prompt = str(self.config.get('segmentation_positive_prompt', ''))
        segmentation_negative_prompt = str(self.config.get('segmentation_negative_prompt', ''))
        thickness = float(self.config.get('mask_outline_thickness', 3.0))
        blur = float(self.config.get('mask_blur', 4))
        if thickness < 0:
            raise ValueError('mask_outline_thickness cannot be negative.')
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        self.load_segmentation()
        try:
            self.console_log('Running segdinosam2 segmentation...')
            result = self.segmentation.segment(image, segmentation_positive_prompt, segmentation_negative_prompt, thickness=0)
            masks = getattr(result, 'masks', None)
            if masks is None and isinstance(result, dict):
                masks = result.get('masks')
            if masks is None:
                mask_image = getattr(result, 'mask_image', None)
                if mask_image is None and isinstance(result, dict):
                    mask_image = result.get('mask_image')
                if mask_image is None:
                    raise RuntimeError('segdinosam2 returned no masks.')
                raw_mask = Image.fromarray(np.asarray(mask_image, dtype=np.uint8), 'L')
                count = 1
            else:
                masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask).any()]
                if not masks:
                    raise RuntimeError('segdinosam2 returned no usable masks.')
                combined = np.zeros(image.size[::-1], dtype=bool)
                for mask in masks:
                    if mask.shape != combined.shape:
                        resized = Image.fromarray((mask.astype(np.uint8) * 255), 'L').resize(image.size, Image.Resampling.NEAREST)
                        mask = np.asarray(resized, dtype=np.uint8) > 0
                    combined |= mask
                raw_mask = Image.fromarray((combined.astype(np.uint8) * 255), 'L')
                count = len(masks)
            if raw_mask.size != image.size:
                raw_mask = raw_mask.resize(image.size, Image.Resampling.NEAREST)
            mask = self.apply_mask_adjustments(raw_mask, thickness)
            if blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
            mask_array = np.asarray(mask, dtype=np.uint8)
            self.console_log(f'Segmentation complete • masks={count} • coverage={mask_array.mean() / 255 * 100:.1f}% • thickness={thickness:g} • blur={blur:g}')
            return mask
        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def extract_original_face_hair(self, original_image):
        self.load_segmentation()
        try:
            self.console_log('Segmenting original face and hair...')
            result = self.segmentation.segment(original_image, FACE_RESTORE_POSITIVE_PROMPT, FACE_RESTORE_NEGATIVE_PROMPT, thickness=0)
            masks = getattr(result, 'masks', None)
            if masks is None and isinstance(result, dict):
                masks = result.get('masks')
            if masks is None:
                mask_image = getattr(result, 'mask_image', None)
                if mask_image is None and isinstance(result, dict):
                    mask_image = result.get('mask_image')
                if mask_image is None:
                    raise RuntimeError('segdinosam2 returned no face/hair mask.')
                raw_mask = Image.fromarray(np.asarray(mask_image, dtype=np.uint8), 'L')
                count = 1
            else:
                valid_masks = []
                for mask in masks:
                    array = np.asarray(mask).astype(bool)
                    if array.ndim > 2:
                        array = np.squeeze(array)
                    if array.ndim != 2 or not np.any(array):
                        continue
                    valid_masks.append(array)
                if not valid_masks:
                    raise RuntimeError('segdinosam2 returned no usable face/hair masks.')
                combined = np.zeros(original_image.size[::-1], dtype=bool)
                for mask in valid_masks:
                    if mask.shape != combined.shape:
                        resized = Image.fromarray((mask.astype(np.uint8) * 255), 'L').resize(original_image.size, Image.Resampling.NEAREST)
                        mask = np.asarray(resized, dtype=np.uint8) > 0
                    combined |= mask
                raw_mask = Image.fromarray((combined.astype(np.uint8) * 255), 'L')
                count = len(valid_masks)
            if raw_mask.size != original_image.size:
                raw_mask = raw_mask.resize(original_image.size, Image.Resampling.NEAREST)
            mask_array = np.asarray(raw_mask, dtype=np.uint8)
            area = np.count_nonzero(mask_array)
            if area < 100:
                raise RuntimeError('No usable face/hair area was detected.')
            self.console_log(f'Original face/hair segmentation complete • masks={count} • coverage={mask_array.mean() / 255 * 100:.1f}%')
            return raw_mask
        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def paste_original_face_hair(self, original_image, output_image):
        original_image = original_image.convert('RGB')
        output_image = output_image.convert('RGB')
        face_hair_mask = self.extract_original_face_hair(original_image)
        if output_image.size != original_image.size:
            original_for_output = original_image.resize(output_image.size, Image.Resampling.LANCZOS)
            face_hair_mask = face_hair_mask.resize(output_image.size, Image.Resampling.NEAREST)
        else:
            original_for_output = original_image
        mask_array = np.asarray(face_hair_mask, dtype=np.uint8)
        if np.count_nonzero(mask_array) < 100:
            raise RuntimeError('No usable original face/hair area was detected.')
        result = Image.composite(original_for_output, output_image, face_hair_mask).convert('RGB')
        self.console_log('Original face and hair pasted onto output')
        return result

    def append_gender_prompts(self, positive_prompt, negative_prompt, selected_gender):
        gender_positive = {'male': 'male character', 'female': 'female character', 'neutral': ''}
        gender_negative = {'male': '', 'female': '', 'neutral': ''}
        positive_append = gender_positive.get(selected_gender, '')
        negative_append = gender_negative.get(selected_gender, '')
        positive_prompt = positive_prompt.strip()
        negative_prompt = negative_prompt.strip()
        if positive_append:
            positive_prompt = f'{positive_append}, {positive_prompt}' if positive_prompt else positive_append
        if negative_append:
            negative_prompt = f'{negative_append}, {negative_prompt}' if negative_prompt else negative_append
        return positive_prompt, negative_prompt

    def append_classification_prompts(self, positive_prompt, negative_prompt, classification):
        style_positive = {REAL: REAL, ANIME: ANIME, THREE_D: THREE_D, CARTOON: CARTOON}
        style_negative = {REAL: f'{ANIME}, {CARTOON}, {THREE_D}', ANIME: f'{REAL}, {THREE_D}, {CARTOON}', THREE_D: f'{REAL}, {ANIME}, {CARTOON}', CARTOON: f'{REAL}, {ANIME}, {THREE_D}'}
        positive_append = style_positive.get(classification, classification)
        negative_append = style_negative.get(classification, '')
        positive_prompt = positive_prompt.strip()
        negative_prompt = negative_prompt.strip()
        if positive_append:
            positive_prompt = f'{positive_append}, {positive_prompt}' if positive_prompt else positive_append
        if negative_append:
            negative_prompt = f'{negative_append}, {negative_prompt}' if negative_prompt else negative_append
        return positive_prompt, negative_prompt

    def generate(self):
        if self.processing:
            return
        if self.model_loading or self.segmentation_loading:
            messagebox.showwarning('Models Loading', 'Please wait until the models finish loading.')
            return
        if self.classification_loading:
            messagebox.showwarning('Image Classification', 'Please wait until the image class and gender are detected.')
            return
        if not self.models_ready:
            messagebox.showwarning('Models Not Ready', 'Please wait until the models finish loading.')
            return
        if self.original_image is None:
            messagebox.showwarning('No Image', 'Please upload an image first.')
            return
        if self.mask_image is None:
            messagebox.showwarning('Mask Not Ready', 'Please click RELOAD MASK first.')
            return
        if self.sd_input_image is None:
            messagebox.showwarning('Input Image Not Ready', 'Please click RELOAD MASK first.')
            return
        try:
            self.sync_config()
        except Exception as e:
            messagebox.showerror('Configuration Error', str(e))
            return
        self.processing = True
        self.output_image = None
        self.generate_btn.configure(state='disabled')
        self.upload_btn.configure(state='disabled')
        self.reload_mask_btn.configure(state='disabled')
        self.save_btn.configure(state='disabled')
        self.model_menu.configure(state='disabled')
        self.image_class_menu.configure(state='disabled')
        self.gender_menu.configure(state='disabled')
        self.start_time = time.time()
        self.show_output()
        self.console_log('Starting generation from processed input image...')
        threading.Thread(target=self.worker, args=(self.sd_input_image.copy(), self.current_model_name), daemon=True).start()

    def classify_input_image(self, file_path):
        return imgclass.classify_image(file_path)

    def composite_mask(self, base, generated, mask):
        if generated.size != base.size:
            generated = generated.resize(base.size, Image.LANCZOS)
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.NEAREST)
        return Image.composite(generated, base, mask).convert('RGB')

    def worker(self, source, model_name):
        try:
            original_source = self.cropped_original_image.copy() if self.cropped_original_image is not None else source.copy()
            selected_class = str(self.image_class_var.get()).strip()
            if not selected_class:
                selected_class = REAL
            selected_gender = str(self.gender_var.get()).strip().lower()
            if selected_gender not in ('male', 'female', 'neutral'):
                selected_gender = 'neutral'
            classification = self.classification_result or {}
            best_probability = float(classification.get('best_probability', 0.0))
            detected_class = str(classification.get('best_class', 'unknown'))
            gender_result = self.gender_result or ('neutral', 0.0)
            detected_gender = str(gender_result[0]).strip().lower()
            gender_confidence = float(gender_result[1])
            self.console_log(f'Overall: image_class={selected_class.upper()} • gender={selected_gender.upper()}')
            self.console_log(f'Detected image class: {detected_class.upper()} ({best_probability * 100:.1f}%)')
            self.console_log(f'Detected gender: {detected_gender.upper()} ({gender_confidence:.1f}%)')
            steps = int(self.config.get('steps', 50))
            cfg = float(self.config.get('cfg', 7.5))
            strength = float(self.config.get('strength', 0.99))
            seed = int(self.config.get('seed', -1))
            guidance_rescale = float(self.config.get('guidance_rescale', 0.0))
            positive_prompt = str(self.config.get('positive_prompt', ''))
            negative_prompt = str(self.config.get('negative_prompt', ''))
            positive_prompt, negative_prompt = self.append_gender_prompts(positive_prompt, negative_prompt, selected_gender)
            positive_prompt, negative_prompt = self.append_classification_prompts(positive_prompt, negative_prompt, selected_class)
            self.console_log(f'Actual positive prompt: {positive_prompt}')
            self.console_log(f'Actual negative prompt: {negative_prompt}')
            self.console_log(f'Generating with selected class {selected_class.upper()} and gender {selected_gender.upper()}')
            image = source.copy()
            mask = self.mask_image.copy() if self.mask_image is not None else None
            self.console_log(f'Using processed input image for Stable Diffusion • {image.width}x{image.height}')
            if mask is None:
                raise RuntimeError('No segmentation mask is available. Please click RELOAD MASK first.')
            if mask.size != image.size:
                raise RuntimeError(f'Processed input and cached mask size do not match: input={image.size}, mask={mask.size}. Please click RELOAD MASK.')
            if np.asarray(mask, dtype=np.uint8).max() < 10:
                raise RuntimeError('No selected segmentation area was detected by segdinosam2.')
            self.console_log(f'Using cached segdinosam2 mask • {mask.width}x{mask.height} • no segmentation rerun')
            w, h = self.generation_size(*image.size)
            init = image.resize((w, h), Image.LANCZOS)
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
            if seed == -1:
                seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            if self.pipe is None:
                raise RuntimeError('Selected diffusion model is not loaded.')
            self.console_log(f'Generation seed: {seed}')
            self.console_log(f'Preparing Stable Diffusion inpainting • {w}x{h}')
            self.restore_diffusion_model()

            def progress(pipe, step_index, timestep, callback_kwargs):
                step = step_index + 1
                elapsed = time.time() - self.start_time
                rate = elapsed / step
                remaining = (steps - step) * rate
                percent = int(step / steps * 100)
                self.console_log(f'{percent:3d}% | {step}/{steps} | [{self.time_text(elapsed)}<{self.time_text(remaining)}] | {model_name}')
                return callback_kwargs

            self.console_log(f'Generating {w}x{h} with {model_name}...')
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            result = self.pipe(prompt=positive_prompt, negative_prompt=negative_prompt, image=init, mask_image=mask, num_inference_steps=steps, guidance_scale=cfg, strength=strength, generator=generator, width=w, height=h, callback_on_step_end=progress, guidance_rescale=guidance_rescale)
            image = result.images[0].convert('RGB')
            image = self.composite_mask(init, image, mask)
            if self.face_restore_var.get():
                try:
                    image = self.paste_original_face_hair(original_source, image)
                except Exception as e:
                    self.console_log(f'Face restore skipped: {e}')
            if self.esrgan_output_var.get():
                image = upscale_img.enhance(image, output=True, logger=self.console_log, output_target=OUTPUT_TARGET)
            image = self.resize_output(image)
            self.output_image = image
            self.output_showing_original = False
            self.after(0, self.show_output)
            self.after(0, lambda: self.save_btn.configure(state='normal'))
            if self.autosave_var.get():
                self.after(0, self.save)
            elapsed = time.time() - self.start_time
            self.console_log(f'Generation complete • {image.width}x{image.height} • {self.time_text(elapsed)}')
        except Exception as e:
            self.console_log(f'Generation error: {e}')
            self.after(0, lambda err=str(e): messagebox.showerror('Generation Error', err))
        finally:
            self.processing = False
            self.after(0, self.update_generate_state)
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.segmentation_loading else 'disabled'))
            self.after(0, lambda: self.reset_crop_btn.configure(state='normal' if self.original_image is not None and self.crop_box is not None else 'disabled'))
            self.after(0, lambda: self.upload_btn.configure(state='normal'))
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.cleanup_gpu()
            self.offload_diffusion_model()

    def generation_size(self, width, height):
        longest = max(width, height)
        if longest <= MAX_SIDE:
            w = max(64, (width // 8) * 8)
            h = max(64, (height // 8) * 8)
        else:
            scale = MAX_SIDE / longest
            w = max(64, (int(round(width * scale)) // 8) * 8)
            h = max(64, (int(round(height * scale)) // 8) * 8)
        return w, h

    def time_text(self, seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    def get_effective_crop_box(self):
        if self.crop_box is None:
            return 0.0, 0.0, 1.0, 1.0
        return self.crop_box

    def get_cropped_image(self, image):
        image = image.convert('RGB')
        if self.crop_box is None:
            return image.copy()
        width, height = image.size
        left, top, right, bottom = self.crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        return image.crop((x1, y1, x2, y2))

    def clamp_crop_box(self, box):
        left, top, right, bottom = [float(value) for value in box]
        minimum = 0.02
        left = max(0.0, min(1.0 - minimum, left))
        top = max(0.0, min(1.0 - minimum, top))
        right = max(minimum, min(1.0, right))
        bottom = max(minimum, min(1.0, bottom))
        if right - left < minimum:
            if left + minimum <= 1.0:
                right = left + minimum
            else:
                left = max(0.0, right - minimum)
        if bottom - top < minimum:
            if top + minimum <= 1.0:
                bottom = top + minimum
            else:
                top = max(0.0, bottom - minimum)
        return left, top, right, bottom

    def crop_handle_size(self):
        return 11

    def crop_handle_points(self, display):
        x, y, width, height = display
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return {'nw': (x1, y1), 'n': (xm, y1), 'ne': (x2, y1), 'w': (x1, ym), 'e': (x2, ym), 'sw': (x1, y2), 's': (xm, y2), 'se': (x2, y2)}

    def get_crop_handle(self, event):
        if self.input_display_info is None:
            return None
        size = self.crop_handle_size() + 5
        for handle, point in self.crop_handle_points(self.input_display_info).items():
            if abs(event.x - point[0]) <= size and abs(event.y - point[1]) <= size:
                return handle
        return None

    def start_crop(self, event):
        if self.original_image is None or self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.input_display_info is None:
            return
        handle = self.get_crop_handle(event)
        if handle is None:
            return
        self.active_crop_handle = handle
        self.crop_box_start = self.get_effective_crop_box()
        self.cropping = True

    def update_crop_selection(self, event):
        if not self.cropping or self.input_display_info is None or self.crop_box_start is None:
            return
        x, y, width, height = self.input_display_info
        px = max(x, min(x + width, event.x))
        py = max(y, min(y + height, event.y))
        nx = (px - x) / width
        ny = (py - y) / height
        left, top, right, bottom = self.crop_box_start
        if self.active_crop_handle in ('nw', 'w', 'sw'):
            left = nx
        if self.active_crop_handle in ('ne', 'e', 'se'):
            right = nx
        if self.active_crop_handle in ('nw', 'n', 'ne'):
            top = ny
        if self.active_crop_handle in ('sw', 's', 'se'):
            bottom = ny
        if self.active_crop_handle in ('nw', 'w', 'sw') and left >= right:
            left = right - 0.02
        if self.active_crop_handle in ('ne', 'e', 'se') and right <= left:
            right = left + 0.02
        if self.active_crop_handle in ('nw', 'n', 'ne') and top >= bottom:
            top = bottom - 0.02
        if self.active_crop_handle in ('sw', 's', 'se') and bottom <= top:
            bottom = top + 0.02
        self.crop_box = self.clamp_crop_box((left, top, right, bottom))
        self.draw_crop_overlay()

    def finish_crop(self, event):
        if not self.cropping:
            return
        self.cropping = False
        self.active_crop_handle = None
        self.crop_box_start = None
        box = self.clamp_crop_box(self.crop_box if self.crop_box is not None else (0.0, 0.0, 1.0, 1.0))
        if box[0] <= 0.005 and box[1] <= 0.005 and box[2] >= 0.995 and box[3] >= 0.995:
            self.crop_box = None
        else:
            self.crop_box = box
        self.cropped_original_image = self.get_cropped_image(self.original_image)
        self.mask_image = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False
        self.reload_mask_btn.configure(state='normal')
        self.reset_crop_btn.configure(state='normal' if self.crop_box is not None else 'disabled')
        self.generate_btn.configure(state='disabled')
        self.console_log('Crop changed • automatic segmentation not restarted • click RELOAD MASK to apply the crop')
        self.show_input()
        self.show_output()

    def reset_crop(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None or self.crop_box is None:
            return
        self.crop_box = None
        self.cropped_original_image = self.original_image.copy()
        self.mask_image = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False
        self.reload_mask_btn.configure(state='normal')
        self.reset_crop_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.console_log('Crop reset • automatic segmentation not restarted • click RELOAD MASK to apply the full image')
        self.show_input()
        self.show_output()

    def compose_mask_for_display(self, image, fitted_size):
        if self.mask_image is None:
            return None
        display_mask = Image.new('L', image.size, 0)
        crop_box = self.get_effective_crop_box()
        width, height = image.size
        x1 = max(0, min(width - 1, int(round(crop_box[0] * width))))
        y1 = max(0, min(height - 1, int(round(crop_box[1] * height))))
        x2 = max(x1 + 1, min(width, int(round(crop_box[2] * width))))
        y2 = max(y1 + 1, min(height, int(round(crop_box[3] * height))))
        crop_size = (x2 - x1, y2 - y1)
        mask = self.mask_image.resize(crop_size, Image.Resampling.NEAREST)
        display_mask.paste(mask, (x1, y1))
        return display_mask.resize(fitted_size, Image.Resampling.NEAREST)

    def draw_crop_overlay(self):
        if self.input_display_info is None:
            return
        self.input_canvas.delete('crop_overlay')
        x, y, width, height = self.input_display_info
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        self.input_canvas.create_rectangle(x1, y1, x2, y2, outline='#FFFFFF', width=3, dash=(8, 5), tags='crop_overlay')
        handle_size = self.crop_handle_size()
        for point in self.crop_handle_points(self.input_display_info).values():
            hx, hy = point
            self.input_canvas.create_rectangle(hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')

    def show_input(self):
        if not hasattr(self, 'input_canvas'):
            return
        self.input_canvas.delete('all')
        image = self.input_image if self.input_image is not None else self.original_image
        if image is None:
            self.input_display_info = None
            canvas_w = max(1, self.input_canvas.winfo_width())
            canvas_h = max(1, self.input_canvas.winfo_height())
            self.input_canvas.create_text(canvas_w // 2, canvas_h // 2, text='Upload Image', fill='#888888', font=('Segoe UI', 22))
            self.reset_crop_btn.configure(state='disabled')
            return
        canvas_w = max(1, self.input_canvas.winfo_width())
        canvas_h = max(1, self.input_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.input_display_info = (x, y, fitted.width, fitted.height)
        if self.mask_var.get() and self.mask_image is not None:
            display_mask = self.compose_mask_for_display(image, fitted.size)
            overlay = Image.new('RGBA', fitted.size, (0, 0, 0, 0))
            red = Image.new('RGBA', fitted.size, (255, 0, 0, 90))
            overlay = Image.composite(red, overlay, display_mask)
            fitted = Image.alpha_composite(fitted.convert('RGBA'), overlay).convert('RGB')
        self.input_photo = ImageTk.PhotoImage(fitted)
        self.input_canvas.create_image(x, y, anchor='nw', image=self.input_photo)
        self.draw_crop_overlay()
        self.reset_crop_btn.configure(state='normal' if self.original_image is not None and self.crop_box is not None and not self.processing and not self.segmentation_loading and not self.classification_loading else 'disabled')

    def show_output(self):
        if not hasattr(self, 'output_canvas'):
            return
        self.output_canvas.delete('all')
        image = self.sd_input_image if self.output_showing_original else self.output_image
        if image is None:
            canvas_w = max(1, self.output_canvas.winfo_width())
            canvas_h = max(1, self.output_canvas.winfo_height())
            loading = self.processing or self.model_loading or self.segmentation_loading or self.classification_loading
            label = 'Loading. Please wait... See on console' if loading else 'Output Image'
            self.output_canvas.create_text(canvas_w // 2, canvas_h // 2, text=label, fill='#888888', font=('Segoe UI', 18))
            return
        canvas_w = max(1, self.output_canvas.winfo_width())
        canvas_h = max(1, self.output_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        self.output_photo = ImageTk.PhotoImage(fitted)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.output_canvas.create_image(x, y, anchor='nw', image=self.output_photo)

    def switch_output_image(self):
        if self.output_image is None and self.sd_input_image is None:
            return
        self.output_showing_original = not self.output_showing_original
        self.show_output()

    def save(self):
        if self.output_image is None:
            return
        path = filedialog.asksaveasfilename(title='Save generated image', defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg *.jpeg'), ('WebP', '*.webp')])
        if not path:
            return
        try:
            output = self.output_image
            suffix = Path(path).suffix.lower()
            if suffix in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            self.console_log(f'Saved image • {path}')
        except Exception as e:
            messagebox.showerror('Save Error', str(e))

if __name__ == '__main__':
    app = App()
    app.mainloop()