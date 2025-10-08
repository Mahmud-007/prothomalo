# ... keep the imports from the earlier GUI ...
import shutil
import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

PHOTOCARD_SCRIPT = "photocard-generator.py"  # change if you keep it elsewhere

class ScrapyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scrapy Runner")
        self.geometry("920x600")
        self.proc = None
        self.project_dir = os.getcwd()

        # Top bar
        top = tk.Frame(self, padx=10, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Scrapy project directory:").pack(side="left")
        self.dir_var = tk.StringVar(value=self.project_dir)
        self.dir_entry = tk.Entry(top, textvariable=self.dir_var, width=70)
        self.dir_entry.pack(side="left", padx=6)
        tk.Button(top, text="Browse…", command=self.pick_dir).pack(side="left")

        # Options
        opts = tk.Frame(self, padx=10, pady=5)
        opts.pack(fill="x")
        self.overwrite_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Overwrite output file (-O)", variable=self.overwrite_var).pack(side="left")

        # Crawl buttons
        crawl = tk.LabelFrame(self, text="Crawl", padx=10, pady=10)
        crawl.pack(fill="x", padx=10, pady=(6, 0))
        self.btn_kalbela = tk.Button(crawl, text="Scrap Kalbela", width=18,
                                     command=lambda: self.run_spider("kalbela", "./articles/kalbela.csv"))
        self.btn_prothomalo = tk.Button(crawl, text="Scrap Prothom Alo", width=18,
                                        command=lambda: self.run_spider("prothomalo", "./articles/prothomalo.csv"))
        self.btn_stop = tk.Button(crawl, text="Stop", width=10, command=self.stop_proc, state="disabled")
        self.btn_kalbela.pack(side="left", padx=4)
        self.btn_prothomalo.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=12)

        # Photocard buttons
        photo = tk.LabelFrame(self, text="Photocard", padx=10, pady=10)
        photo.pack(fill="x", padx=10, pady=6)
        self.btn_photo_kalbela = tk.Button(photo, text="Generate Photocard (Kalbela)", width=28,
                                           command=lambda: self.run_photocard("kalbela"))
        self.btn_photo_prothomalo = tk.Button(photo, text="Generate Photocard (Prothom Alo)", width=28,
                                              command=lambda: self.run_photocard("prothomalo"))
        self.btn_photo_kalbela.pack(side="left", padx=4)
        self.btn_photo_prothomalo.pack(side="left", padx=4)

        # Log area
        self.log = scrolledtext.ScrolledText(self, height=22, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=5)
        self.write_log("Ready.\n")

    # --- unchanged helpers (pick_dir, write_log, set_buttons_running) ---

    def pick_dir(self):
        selected = filedialog.askdirectory(initialdir=self.project_dir)
        if selected:
            self.project_dir = selected
            self.dir_var.set(selected)

    def write_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")
        self.log.update_idletasks()

    def set_buttons_running(self, running: bool):
        state_main = "disabled" if running else "normal"
        for btn in (self.btn_kalbela, self.btn_prothomalo,
                    self.btn_photo_kalbela, self.btn_photo_prothomalo):
            btn.config(state=state_main)
        self.btn_stop.config(state="normal" if running else "disabled")

    # --- crawl ---
    def run_spider(self, spider_name: str, out_relpath: str):
        if self.proc is not None:
            messagebox.showwarning("Already Running", "A process is already in progress.")
            return

        project_dir = self.dir_var.get().strip() or os.getcwd()
        if not os.path.isdir(project_dir):
            messagebox.showerror("Invalid Directory", "Please select a valid Scrapy project directory.")
            return

        out_path = os.path.normpath(os.path.join(project_dir, out_relpath))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cmd = [sys.executable, "-m", "scrapy", "crawl", spider_name]
        if self.overwrite_var.get():
            cmd += ["-O", out_path]
        else:
            cmd += ["-o", out_path]

        self._spawn(cmd, project_dir)

    # --- photocard ---
    def run_photocard(self, site: str):
        if self.proc is not None:
            messagebox.showwarning("Already Running", "A process is already in progress.")
            return

        project_dir = self.dir_var.get().strip() or os.getcwd()
        if not os.path.isdir(project_dir):
            messagebox.showerror("Invalid Directory", "Please select a valid project directory.")
            return

        script_path = os.path.join(project_dir, PHOTOCARD_SCRIPT)
        if not os.path.isfile(script_path):
            messagebox.showerror("Script Not Found",
                                 f"Could not find {PHOTOCARD_SCRIPT} in:\n{project_dir}")
            return

        # Keep the command exactly `python photocard-generator.py`
        cmd = [sys.executable, PHOTOCARD_SCRIPT]
        # Inject SITE env var so your script chooses the right CSV/OUT paths
        env = os.environ.copy()
        env["SITE"] = site.lower()

        self.write_log(f"\n>>> Running: SITE={env['SITE']} {' '.join(cmd)}\n")
        self.write_log(f">>> Working dir: {project_dir}\n")

        def target():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=project_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
                self.set_buttons_running(True)
                for line in self.proc.stdout:
                    self.write_log(line)
                code = self.proc.wait()
                self.write_log(f"\n>>> Process exited with code {code}\n")
            except Exception as e:
                self.write_log(f"\n[ERROR] {e}\n")
            finally:
                self.proc = None
                self.set_buttons_running(False)

        threading.Thread(target=target, daemon=True).start()

    def _spawn(self, cmd, cwd):
        self.write_log(f"\n>>> Running: {' '.join(cmd)}\n")
        self.write_log(f">>> Working dir: {cwd}\n")

        def target():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                self.set_buttons_running(True)
                for line in self.proc.stdout:
                    self.write_log(line)
                code = self.proc.wait()
                self.write_log(f"\n>>> Process exited with code {code}\n")
            except Exception as e:
                self.write_log(f"\n[ERROR] {e}\n")
            finally:
                self.proc = None
                self.set_buttons_running(False)

        threading.Thread(target=target, daemon=True).start()

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self.write_log(">>> Stopping process…\n")
            try:
                self.proc.terminate()
            except Exception as e:
                self.write_log(f"[ERROR] terminate() failed: {e}\n")
        else:
            self.write_log(">>> No running process.\n")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
    app = ScrapyGUI()
    app.mainloop()
