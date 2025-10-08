import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# --- Config you may tweak ---
DEFAULT_OUTPUT_DIR = "./articles"  # relative to the Scrapy project dir
# ----------------------------

class ScrapyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scrapy Runner")
        self.geometry("820x520")
        self.proc = None
        self.project_dir = os.getcwd()

        # Top bar: project dir chooser
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
        self.overwrite_var = tk.BooleanVar(value=True)   # use -O to overwrite
        tk.Checkbutton(opts, text="Overwrite output file (-O)", variable=self.overwrite_var).pack(side="left")

        # Buttons
        btns = tk.Frame(self, padx=10, pady=10)
        btns.pack(fill="x")
        self.btn_kalbela = tk.Button(btns, text="Scrap Kalbela", width=18,
                                     command=lambda: self.run_spider("kalbela", f"{DEFAULT_OUTPUT_DIR}/kalbela.csv"))
        self.btn_prothomalo = tk.Button(btns, text="Scrap Prothom Alo", width=18,
                                        command=lambda: self.run_spider("prothomalo", f"{DEFAULT_OUTPUT_DIR}/prothomalo.csv"))
        self.btn_stop = tk.Button(btns, text="Stop", width=10, command=self.stop_proc, state="disabled")

        self.btn_kalbela.pack(side="left", padx=4)
        self.btn_prothomalo.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=12)

        # Log area
        self.log = scrolledtext.ScrolledText(self, height=22, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=5)
        self.write_log("Ready.\n")

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
        self.btn_kalbela.config(state=state_main)
        self.btn_prothomalo.config(state=state_main)
        self.btn_stop.config(state="normal" if running else "disabled")

    def run_spider(self, spider_name: str, output_relpath: str):
        if self.proc is not None:
            messagebox.showwarning("Already Running", "A crawl is already in progress.")
            return

        project_dir = self.dir_var.get().strip() or os.getcwd()
        if not os.path.isdir(project_dir):
            messagebox.showerror("Invalid Directory", "Please select a valid Scrapy project directory.")
            return

        # Ensure output dir exists (relative to project dir)
        out_path = os.path.normpath(os.path.join(project_dir, output_relpath))
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # Build command: use the same Python that's running this GUI
        # `python -m scrapy` is reliable across platforms/venvs.
        cmd = [sys.executable, "-m", "scrapy", "crawl", spider_name]
        if self.overwrite_var.get():
            cmd += ["-O", out_path]   # overwrite
        else:
            cmd += ["-o", out_path]   # append/new

        self.write_log(f"\n>>> Running: {' '.join(cmd)}\n")
        self.write_log(f">>> Working dir: {project_dir}\n")

        # Launch subprocess in a background thread to keep UI responsive
        def target():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=project_dir,
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
                # cross-platform terminate
                self.proc.terminate()
            except Exception as e:
                self.write_log(f"[ERROR] terminate() failed: {e}\n")

        else:
            self.write_log(">>> No running process.\n")

if __name__ == "__main__":
    # Nice default: start in the folder where you double-clicked the EXE/py
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
    app = ScrapyGUI()
    app.mainloop()
