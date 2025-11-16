#!/usr/bin/env python3
"""
Simple desktop GUI wrapper around lecture_pipeline.py.

Mac-friendly drag-and-drop interface:
  • Drop a lecture directory (or a single audio/video file) to run the pipeline.
  • Provide/OpenAI API key in a password-style field before launching audio jobs.
  • View live logs and stop the run when needed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD
except Exception:  # pragma: no cover - optional dependency
    TkinterDnD = None
    DND_FILES = None
    DND_TEXT = None


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = SCRIPT_DIR / "lecture_pipeline.py"
MEDIA_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".webm",
    ".mkv",
    ".mp4",
    ".mov",
    ".avi",
    ".m4v",
}


def format_command(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


class LectureToolsApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Lecture Tools")
        self.master.geometry("760x520")
        self.master.minsize(640, 420)

        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.api_key_var = tk.StringVar()

        existing_key = os.environ.get("OPENAI_API_KEY")
        if existing_key:
            self.api_key_var.set(existing_key)

        self.current_process: Optional[subprocess.Popen[str]] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        self.dnd_supported = TkinterDnD is not None
        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.master, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        path_label = ttk.Label(main, text="Lecture directory or media file:")
        path_label.pack(anchor="w")

        path_row = ttk.Frame(main)
        path_row.pack(fill=tk.X, pady=(4, 8))

        self.path_entry = ttk.Entry(path_row, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_dir = ttk.Button(path_row, text="Browse Folder…", command=self._pick_directory)
        browse_dir.pack(side=tk.LEFT, padx=(8, 0))

        browse_file = ttk.Button(path_row, text="Browse File…", command=self._pick_file)
        browse_file.pack(side=tk.LEFT, padx=(4, 0))

        drop_text = "Drop folders or audio/video files here"
        if not self.dnd_supported:
            drop_text += " (install tkinterdnd2 for drag-and-drop)"
        self.drop_area = tk.Label(
            main,
            text=drop_text,
            relief=tk.RIDGE,
            anchor="center",
            padx=24,
            pady=24,
            bg="#f4f4f4",
        )
        self.drop_area.pack(fill=tk.X, pady=(0, 12))

        api_label = ttk.Label(main, text="OpenAI API key:")
        api_label.pack(anchor="w")

        api_row = ttk.Frame(main)
        api_row.pack(fill=tk.X, pady=(4, 12))

        self.api_entry = ttk.Entry(api_row, textvariable=self.api_key_var, show="•")
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        clear_api = ttk.Button(api_row, text="Clear", command=lambda: self.api_key_var.set(""))
        clear_api.pack(side=tk.LEFT, padx=(6, 0))

        actions = ttk.Frame(main)
        actions.pack(fill=tk.X)

        self.run_button = ttk.Button(actions, text="Run Lecture Pipeline", command=self._launch_pipeline)
        self.run_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop_pipeline, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        status_label = ttk.Label(actions, textvariable=self.status_var, anchor="w")
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(16, 0))

        log_label = ttk.Label(main, text="Logs:")
        log_label.pack(anchor="w", pady=(12, 0))

        self.log_widget = scrolledtext.ScrolledText(main, height=12, state=tk.DISABLED)
        self.log_widget.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        if self.dnd_supported:
            self._enable_drag_and_drop()

    def _enable_drag_and_drop(self) -> None:
        # Create DnD-enabled root if available
        if TkinterDnD and not isinstance(self.master, TkinterDnD.Tk):
            messagebox.showinfo(
                "Drag-and-drop disabled",
                "Drag-and-drop only works when the app is launched via lecture_gui.py.",
            )
            return

        widgets = [self.path_entry, self.drop_area]
        for widget in widgets:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._handle_path_drop)

        if DND_TEXT:
            self.api_entry.drop_target_register(DND_TEXT)
            self.api_entry.dnd_bind("<<Drop>>", self._handle_api_drop)

    def _pick_directory(self) -> None:
        directory = filedialog.askdirectory(parent=self.master, title="Select lecture directory")
        if directory:
            self._set_target_path(Path(directory))

    def _pick_file(self) -> None:
        file_path = filedialog.askopenfilename(
            parent=self.master,
            title="Select audio or video file",
            filetypes=[("Media files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.mkv *.mp4 *.mov *.avi *.m4v"), ("All files", "*.*")],
        )
        if file_path:
            self._set_target_path(Path(file_path))

    def _handle_path_drop(self, event) -> None:  # type: ignore[no-untyped-def]
        data = event.data
        try:
            paths = self.master.tk.splitlist(data)
        except Exception:
            paths = [data]
        if not paths:
            return
        self._set_target_path(Path(paths[0]))

    def _handle_api_drop(self, event) -> None:  # type: ignore[no-untyped-def]
        data = (event.data or "").strip()
        if not data:
            return
        cleaned = data
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.strip().replace("\n", "")
        if cleaned:
            self.api_key_var.set(cleaned)
            self._append_log("Updated API key from drop.\n")

    def _set_target_path(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            messagebox.showerror("Path not found", f"{resolved} does not exist.")
            return

        target = resolved
        if resolved.is_file():
            if resolved.suffix.lower() not in MEDIA_SUFFIXES:
                self._append_log(
                    f"Selected file {resolved.name} is not a typical audio/video file. "
                    "Using its parent directory.\n"
                )
            target = resolved.parent
        self.path_var.set(str(target))
        self._append_log(f"Target path set to: {target}\n")

    def _launch_pipeline(self) -> None:
        if self.running:
            messagebox.showinfo("Busy", "The pipeline is already running.")
            return

        path_text = self.path_var.get().strip()
        if not path_text:
            messagebox.showwarning("Missing path", "Please select or drop a lecture directory first.")
            return

        target_dir = Path(path_text).expanduser()
        if not target_dir.exists() or not target_dir.is_dir():
            messagebox.showerror("Invalid directory", f"{target_dir} is not a valid directory.")
            return

        if not PIPELINE_PATH.exists():
            messagebox.showerror("Missing pipeline", f"Could not find lecture_pipeline.py at {PIPELINE_PATH}")
            return

        self.running = True
        self.status_var.set("Running…")
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._append_log(f"Starting pipeline in {target_dir}\n")

        self.worker_thread = threading.Thread(target=self._run_pipeline, args=(target_dir,), daemon=True)
        self.worker_thread.start()

    def _run_pipeline(self, target_dir: Path) -> None:
        env = os.environ.copy()
        api_key = self.api_key_var.get().strip()
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        cmd = [sys.executable, str(PIPELINE_PATH), str(target_dir)]
        self._async(lambda: self._append_log(f"Command: {format_command(cmd)}\n"))

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._async(lambda: self._finish_run(False, f"Failed to start pipeline: {exc}\n"))
            return

        self.current_process = process

        try:
            assert process.stdout is not None
            for line in process.stdout:
                self._async(lambda text=line: self._append_log(text))
            rc = process.wait()
        finally:
            self.current_process = None

        success = rc == 0
        tail = "[success]\n" if success else f"[failed with code {rc}]\n"
        self._async(lambda: self._finish_run(success, tail))

    def _finish_run(self, success: bool, message: str) -> None:
        self._append_log(message)
        self.status_var.set("Idle" if success else "Failed")
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.running = False
        if not success:
            messagebox.showerror("Lecture pipeline failed", "Check the logs for details.")

    def _stop_pipeline(self) -> None:
        proc = self.current_process
        if proc and proc.poll() is None:
            proc.terminate()
            self._append_log("Sent termination signal…\n")
            self.status_var.set("Stopping…")
        else:
            self.stop_button.config(state=tk.DISABLED)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _async(self, func) -> None:  # type: ignore[no-untyped-def]
        self.master.after(0, func)


def main() -> None:
    root: tk.Tk
    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = LectureToolsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
