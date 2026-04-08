import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from src.autoclicker import run_scanner


class App:
    def __init__(self, root):
        self.root = root
        root.title("OCR Autoclicker GUI")

        tk.Label(root, text="Target (word or regex):").grid(row=0, column=0, sticky='e')
        self.target_var = tk.StringVar(value="Click")
        tk.Entry(root, textvariable=self.target_var, width=40).grid(row=0, column=1, columnspan=3, sticky='we')

        self.regex_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Regex", variable=self.regex_var).grid(row=0, column=4, sticky='w')

        tk.Label(root, text="Region L T W H:").grid(row=1, column=0, sticky='e')
        self.left_var = tk.StringVar()
        self.top_var = tk.StringVar()
        self.w_var = tk.StringVar()
        self.h_var = tk.StringVar()
        tk.Entry(root, width=6, textvariable=self.left_var).grid(row=1, column=1)
        tk.Entry(root, width=6, textvariable=self.top_var).grid(row=1, column=2)
        tk.Entry(root, width=6, textvariable=self.w_var).grid(row=1, column=3)
        tk.Entry(root, width=6, textvariable=self.h_var).grid(row=1, column=4)

        tk.Label(root, text="Interval (s):").grid(row=2, column=0, sticky='e')
        self.interval_var = tk.DoubleVar(value=0.5)
        tk.Entry(root, textvariable=self.interval_var, width=8).grid(row=2, column=1)

        tk.Label(root, text="Confidence:").grid(row=2, column=2, sticky='e')
        self.conf_var = tk.DoubleVar(value=60.0)
        tk.Entry(root, textvariable=self.conf_var, width=8).grid(row=2, column=3)

        self.binarize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="Binarize", variable=self.binarize_var).grid(row=2, column=4, sticky='w')

        tk.Label(root, text="Threshold:").grid(row=3, column=0, sticky='e')
        self.threshold_var = tk.IntVar(value=150)
        tk.Entry(root, textvariable=self.threshold_var, width=8).grid(row=3, column=1)

        tk.Label(root, text="Tesseract cmd:").grid(row=3, column=2, sticky='e')
        self.tess_var = tk.StringVar()
        tk.Entry(root, textvariable=self.tess_var, width=22).grid(row=3, column=3)
        tk.Button(root, text="Browse", command=self.browse_tesseract).grid(row=3, column=4)

        self.start_btn = tk.Button(root, text="Start", command=self.toggle_start)
        self.start_btn.grid(row=4, column=0, pady=6)
        tk.Button(root, text="Quit", command=self.on_quit).grid(row=4, column=1)

        self.log = tk.Text(root, height=12, width=72)
        self.log.grid(row=5, column=0, columnspan=5, pady=6)

        self.thread = None
        self.running_event = None
        self.stop_event = None

    def browse_tesseract(self):
        path = filedialog.askopenfilename(title="Select tesseract.exe", filetypes=[("exe files", "*.exe"), ("All files", "*.*")])
        if path:
            self.tess_var.set(path)

    def append_log(self, msg: str):
        def _append():
            self.log.insert(tk.END, msg + "\n")
            self.log.see(tk.END)
        self.root.after(0, _append)

    def toggle_start(self):
        # If thread running, toggle pause/resume
        if self.thread and self.thread.is_alive():
            if self.running_event and self.running_event.is_set():
                self.running_event.clear()
                self.start_btn.config(text="Resume")
                self.append_log("Paused scanning")
            else:
                self.running_event.set()
                self.start_btn.config(text="Pause")
                self.append_log("Resumed scanning")
            return

        # Start a new scanner thread
        target = self.target_var.get().strip()
        if not target:
            messagebox.showwarning("Validation", "Please set a target word or regex.")
            return

        region = None
        try:
            if self.left_var.get() and self.top_var.get() and self.w_var.get() and self.h_var.get():
                left = int(self.left_var.get()); top = int(self.top_var.get()); w = int(self.w_var.get()); h = int(self.h_var.get())
                region = {"left": left, "top": top, "width": w, "height": h}
        except ValueError:
            messagebox.showwarning("Validation", "Region values must be integers.")
            return

        cfg = dict(
            target=target,
            region=region,
            regex=self.regex_var.get(),
            interval=float(self.interval_var.get()),
            confidence=float(self.conf_var.get()),
            binarize=self.binarize_var.get(),
            threshold=int(self.threshold_var.get()),
            pause=0.05,
            tesseract_cmd=self.tess_var.get() or None,
        )

        self.stop_event = threading.Event()
        self.running_event = threading.Event()
        self.running_event.set()

        def worker():
            run_scanner(cfg['target'], cfg['region'], cfg['regex'], cfg['interval'], cfg['confidence'], cfg['binarize'], cfg['threshold'], cfg['pause'], cfg['tesseract_cmd'], self.running_event, self.stop_event, log_callback=self.append_log)
            self.append_log("Scanner stopped")
            self.start_btn.config(text="Start")

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        self.start_btn.config(text="Pause")
        self.append_log("Started scanner")

    def on_quit(self):
        if self.stop_event:
            self.stop_event.set()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
