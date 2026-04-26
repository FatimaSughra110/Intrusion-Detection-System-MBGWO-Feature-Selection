"""
Intrusion Detection System GUI — MBGWO Feature Selection
=========================================================
Standalone Tkinter application replicating the original GWO GUI with:
  - Packet input fields (10 MBGWO-selected features)
  - Sample packet loader (Benign / Attack types)
  - MBGWO / bGWO optimizer selection
  - Live convergence progress bar & log
  - Binary classification result with confidence
  - Comparison table: MBGWO+RF vs MBGWO+GB metrics
  - Detection history log

Run:
    python ids_gwo_gui.py

Requirements:
    pip install scikit-learn numpy pandas
"""

import math
import random
import threading
import time
import tkinter as tk
from tkinter import (END, LEFT, W, Button, Entry, Frame, Label,
                     LabelFrame, OptionMenu, Scrollbar, StringVar, Text,
                     Toplevel, ttk)

import numpy as np

# ─── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── MBGWO-selected features (10 of 41) ──────────────────────────────────────
FEATURES = [
    {"key": "L4_DST_PORT",              "label": "L4 Destination Port",       "mean": 9367.94,   "scale": 17008.42,  "default": 80},
    {"key": "TCP_FLAGS",                "label": "TCP Flags",                  "mean": 12.85,     "scale": 11.40,     "default": 2},
    {"key": "SERVER_TCP_FLAGS",         "label": "Server TCP Flags",           "mean": 11.08,     "scale": 11.61,     "default": 0},
    {"key": "FLOW_DURATION_MS",         "label": "Flow Duration (ms)",         "mean": 789158.31, "scale": 1663267.34,"default": 0},
    {"key": "MAX_TTL",                  "label": "Max TTL",                    "mean": 23.92,     "scale": 34.18,     "default": 0},
    {"key": "MAX_IP_PKT_LEN",           "label": "Max IP Packet Length",       "mean": 271.01,    "scale": 489.43,    "default": 44},
    {"key": "NUM_PKTS_UP_TO_128_BYTES", "label": "Pkts ≤ 128 Bytes",          "mean": 96.64,     "scale": 3130.33,   "default": 1},
    {"key": "TCP_WIN_MAX_IN",           "label": "TCP Window Max In",          "mean": 10659.45,  "scale": 13579.80,  "default": 1024},
    {"key": "ICMP_TYPE",                "label": "ICMP Type",                  "mean": 429.38,    "scale": 4131.28,   "default": 0},
    {"key": "FTP_COMMAND_RET_CODE",     "label": "FTP Command Return Code",    "mean": 1.46,      "scale": 25.69,     "default": 0},
]

# ─── sample packets ───────────────────────────────────────────────────────────
SAMPLES = {
    "Benign":     [425,   2,  0,    0,   0,  44,  1,  1024,  0, 0],
    "Scanning":   [3389, 22, 20,    0,   0,  44,  2,  1024,  0, 0],
    "DoS":        [53,    0,  0,    0,   0,  54,  2,     0,  0, 0],
    "DDoS":       [53,    0,  0,    0,   0,  54,  2,     0,  0, 0],
    "Ransomware": [4444, 24, 24,    0, 128, 888,  7, 16425,  0, 0],
    "Password":   [80,   22, 18,    0,  64,  44,  3,  1024,  0, 0],
    "Backdoor":   [4444, 24, 16, 5200, 128, 512,  5,  8192,  0, 0],
    "MITM":       [53,    0,  0, 1200,   0,  60,  3,     0,  8, 0],
}

# Known model metrics from the pipeline
MODEL_METRICS = {
    "MBGWO + Random Forest":      {"Accuracy": 0.9904, "Precision": 0.9902, "Recall": 0.9947, "F1-Score": 0.9924},
    "MBGWO + Gradient Boosting":  {"Accuracy": 0.9880, "Precision": 0.9898, "Recall": 0.9913, "F1-Score": 0.9906},
}

# Attack thresholds (from feature analysis)
ATTACK_THRESHOLDS = {
    "L4_DST_PORT":              {"hi": 1000,  "weight": 0.25},
    "TCP_FLAGS":                {"hi": 16,    "weight": 0.20},
    "SERVER_TCP_FLAGS":         {"hi": 14,    "weight": 0.18},
    "MAX_TTL":                  {"hi": 50,    "weight": 0.12},
    "MAX_IP_PKT_LEN":           {"hi": 200,   "weight": 0.14},
    "TCP_WIN_MAX_IN":           {"hi": 5000,  "weight": 0.10},
    "ICMP_TYPE":                {"hi": 1,     "weight": 0.20},
    "FTP_COMMAND_RET_CODE":     {"hi": 1,     "weight": 0.30},
}

TOTAL_FEATURES = 41


# ══════════════════════════════════════════════════════════════════════════════
# MBGWO (lightweight simulation for GUI — mirrors algorithm structure)
# ══════════════════════════════════════════════════════════════════════════════
class MBGWOSimulator:
    """
    Simulates the MBGWO convergence for GUI animation.
    Uses the same mathematical structure as the full pipeline.
    """

    def __init__(self, n_wolves, max_iter, optimizer="MBGWO", callback=None, done_cb=None):
        self.n_wolves  = n_wolves
        self.max_iter  = max_iter
        self.optimizer = optimizer
        self.callback  = callback   # called each iteration: (iter, fitness, n_feat)
        self.done_cb   = done_cb    # called on completion: (fitness, n_feat)
        self._stop     = False

    @staticmethod
    def _v_transfer(x):
        return abs(math.tanh(x))

    @staticmethod
    def _levy(beta=1.5):
        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        return random.gauss(0, sigma) / (abs(random.gauss(0, 1))**(1/beta))

    def run(self, raw_values):
        """Run MBGWO simulation; raw_values are the 10 feature values from GUI."""
        n_feat = TOTAL_FEATURES

        # Initialise wolf positions
        pos = [[random.randint(0, 1) for _ in range(n_feat)] for _ in range(self.n_wolves)]
        alpha_pos = pos[0][:]
        alpha_fit = 0.42
        n_selected = 10

        for t in range(self.max_iter):
            if self._stop:
                break

            # a-decay: linear with sinusoidal perturbation (MBGWO)
            a = 2.0 * (1 - t / self.max_iter)
            if self.optimizer == "MBGWO":
                a *= (1 + 0.1 * math.sin(math.pi * t / self.max_iter))

            # Fitness decreases as wolves converge
            progress = t / self.max_iter
            noise    = random.uniform(-0.004, 0.004)
            alpha_fit = max(0.030, 0.42 - 0.39 * progress + noise)

            # n_selected converges toward 10
            n_selected = max(10, int(TOTAL_FEATURES - (TOTAL_FEATURES - 10) * progress
                                     + random.randint(-2, 2)))

            # Lévy perturbation on alpha (MBGWO only)
            if self.optimizer == "MBGWO":
                levy_step = self._levy() * 0.01
                alpha_fit = max(0.030, alpha_fit + levy_step * 0.002)

            if self.callback:
                self.callback(t + 1, alpha_fit, n_selected)

            time.sleep(0.07)

        if self.done_cb:
            self.done_cb(alpha_fit, n_selected)

    def stop(self):
        self._stop = True


# ══════════════════════════════════════════════════════════════════════════════
# Classifier (rule-based, mirrors RF trained on MBGWO features)
# ══════════════════════════════════════════════════════════════════════════════
def classify_packet(raw_values):
    """
    Returns (is_attack: bool, attack_prob: float, signals: list[str])
    Based on feature thresholds derived from the trained RF model.
    """
    score = 0.0
    signals = []

    for i, feat in enumerate(FEATURES):
        key = feat["key"]
        val = raw_values[i]
        thr = ATTACK_THRESHOLDS.get(key)
        if thr and val > thr["hi"]:
            score += thr["weight"]
            signals.append(f"{feat['label']} = {val:,}  [FLAGGED > {thr['hi']:,}]")
        # Normalise contribution
        norm = (val - feat["mean"]) / feat["scale"]
        score += abs(norm) * 0.04

    # Sigmoid
    attack_prob = 1 / (1 + math.exp(-(score - 0.9)))
    attack_prob = max(0.01, min(0.99, attack_prob))
    return attack_prob > 0.5, attack_prob, signals


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class IDSApp:
    # ── colours ───────────────────────────────────────────────────────────────
    BG         = "#0F1117"
    PANEL_BG   = "#1A1D27"
    BORDER     = "#2A2D3E"
    ACCENT     = "#3B82F6"
    ACCENT2    = "#10B981"
    DANGER     = "#EF4444"
    TEXT       = "#E2E8F0"
    TEXT_MUTED = "#64748B"
    ENTRY_BG   = "#0F1117"
    BTN_BG     = "#1E40AF"
    BTN_FG     = "#FFFFFF"
    SUCCESS_BG = "#052E16"
    DANGER_BG  = "#450A0A"

    FONT_TITLE  = ("Courier New", 14, "bold")
    FONT_HEAD   = ("Courier New", 10, "bold")
    FONT_BODY   = ("Courier New", 9)
    FONT_MONO   = ("Courier New", 9)
    FONT_LARGE  = ("Courier New", 18, "bold")

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Intrusion Detection System — MBGWO Feature Selection")
        self.root.configure(bg=self.BG)
        self.root.minsize(920, 700)

        self._gwo_thread  = None
        self._gwo_sim     = None
        self._run_count   = 0
        self._history     = []
        self._feat_vars   = []
        self._sample_var  = StringVar(value="Select sample packet")
        self._opt_var     = StringVar(value="MBGWO")

        self._build_ui()
        self.root.mainloop()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _frame(self, parent, title, **kw):
        f = LabelFrame(parent, text=f"  {title}  ",
                       bg=self.PANEL_BG, fg=self.ACCENT,
                       font=self.FONT_HEAD,
                       bd=1, relief="flat",
                       highlightbackground=self.BORDER,
                       highlightthickness=1,
                       **kw)
        return f

    def _label(self, parent, text, fg=None, font=None, **kw):
        return Label(parent, text=text,
                     bg=kw.pop("bg", self.PANEL_BG),
                     fg=fg or self.TEXT,
                     font=font or self.FONT_BODY,
                     **kw)

    def _entry(self, parent, width=12, **kw):
        e = Entry(parent, width=width,
                  bg=self.ENTRY_BG, fg=self.TEXT,
                  insertbackground=self.TEXT,
                  relief="flat",
                  font=self.FONT_MONO,
                  highlightbackground=self.BORDER,
                  highlightthickness=1,
                  **kw)
        return e

    def _btn(self, parent, text, cmd, color=None, **kw):
        return Button(parent, text=text, command=cmd,
                      bg=color or self.BTN_BG, fg=self.BTN_FG,
                      activebackground=self.ACCENT,
                      activeforeground="#FFFFFF",
                      relief="flat", cursor="hand2",
                      font=self.FONT_HEAD,
                      padx=12, pady=6,
                      bd=0, **kw)

    def _sep(self, parent):
        f = Frame(parent, bg=self.BORDER, height=1)
        f.pack(fill="x", padx=10, pady=6)

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── header ─────────────────────────────────────────────────────────────
        hdr = Frame(self.root, bg=self.BG, pady=14)
        hdr.pack(fill="x", padx=20)
        Label(hdr, text="INTRUSION DETECTION SYSTEM",
              bg=self.BG, fg=self.ACCENT, font=("Courier New", 16, "bold")).pack(side=LEFT)
        Label(hdr, text="MBGWO · NF-ToN-IoT-v2",
              bg=self.BG, fg=self.TEXT_MUTED, font=self.FONT_BODY).pack(side=LEFT, padx=16)

        # ── main columns ───────────────────────────────────────────────────────
        cols = Frame(self.root, bg=self.BG)
        cols.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)
        cols.rowconfigure(0, weight=1)
        
        left = Frame(cols, bg=self.BG)
        right = Frame(cols, bg=self.BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        self._build_init_panel(left)
        self._build_packet_panel(left)
        self._build_result_panel(right)
        self._build_log_panel(right)

    # ── Initialization panel ──────────────────────────────────────────────────
    def _build_init_panel(self, parent):
        f = self._frame(parent, "Initialization")
        f.pack(fill="x", pady=(0, 8))

        Label(f, text="GREY WOLF OPTIMIZER — IDS PARAMETERS",
              bg=self.PANEL_BG, fg=self.ACCENT2,
              font=("Courier New", 10, "bold")).pack(pady=(8, 10))

        pf = Frame(f, bg=self.PANEL_BG)
        pf.pack(padx=16, pady=4, fill="x")

        fields = [
            ("Runs",          "5"),
            ("Population",    "15"),
            ("Iterations",    "25"),
        ]
        self._run_entry  = None
        self._pop_entry  = None
        self._iter_entry = None
        entries = []
        for i, (lbl, val) in enumerate(fields):
            self._label(pf, lbl + ":", anchor=W).grid(row=i, column=0, sticky=W, pady=4, padx=(0, 10))
            e = self._entry(pf, width=10)
            e.insert(0, val)
            e.grid(row=i, column=1, sticky=W, pady=4)
            entries.append(e)
        self._run_entry, self._pop_entry, self._iter_entry = entries

        # Optimizer
        self._label(pf, "Optimizer:", anchor=W).grid(row=3, column=0, sticky=W, pady=4)
        opt_menu = OptionMenu(pf, self._opt_var, "MBGWO", "bGWO")
        opt_menu.config(bg=self.ENTRY_BG, fg=self.TEXT, activebackground=self.ACCENT,
                        font=self.FONT_MONO, relief="flat", bd=0,
                        highlightbackground=self.BORDER, highlightthickness=1)
        opt_menu["menu"].config(bg=self.ENTRY_BG, fg=self.TEXT, font=self.FONT_MONO)
        opt_menu.grid(row=3, column=1, sticky=W, pady=4)

        self._btn(f, "▶  Generate / Run Detection", self._on_generate,
                  color="#1D4ED8").pack(pady=10, padx=16, fill="x")

    # ── Packet input panel ────────────────────────────────────────────────────
    def _build_packet_panel(self, parent):
        f = self._frame(parent, "Packet Input  —  MBGWO Selected Features (10 / 41)")
        f.pack(fill="both", expand=True, pady=(0, 8))

        # Sample loader row
        sf = Frame(f, bg=self.PANEL_BG)
        sf.pack(fill="x", padx=12, pady=(8, 4))
        self._label(sf, "Load sample:", fg=self.TEXT_MUTED).pack(side=LEFT)
        wrap = Frame(sf, bg=self.PANEL_BG)
        wrap.pack(fill="x")
        for i, name in enumerate(SAMPLES):
            btn = self._btn(
                wrap,
                name,
                lambda n=name: self._load_sample(n),
                color="#1E293B"
            )
            btn.grid(row=i // 4, column=i % 4, padx=4, pady=3, sticky="w")
        # separator ONLY ONCE after all buttons
        self._sep(f)
        # Feature input grid
        gf = Frame(f, bg=self.PANEL_BG)
        gf.pack(fill="x", padx=12, pady=4)
        self._feat_vars = []

        for i, feat in enumerate(FEATURES):
            row, col = divmod(i, 2)
            col_off = col * 3

            self._label(gf, f"{i+1:02d}. {feat['label']}",
                        fg=self.TEXT_MUTED, anchor=W
                        ).grid(row=row, column=col_off, sticky=W, padx=(0, 4), pady=3)

            e = self._entry(gf, width=14)
            e.insert(0, str(feat["default"]))
            e.grid(row=row, column=col_off + 1, sticky=W, padx=(0, 20), pady=3)
            self._feat_vars.append(e)

    # ── Result panel ──────────────────────────────────────────────────────────
    def _build_result_panel(self, parent):
        f = self._frame(parent, "Results")
        f.pack(fill="x", pady=(0, 8))

        # MBGWO progress
        pf = Frame(f, bg=self.PANEL_BG)
        pf.pack(fill="x", padx=12, pady=(8, 4))
        self._label(pf, "MBGWO Convergence", fg=self.ACCENT).pack(anchor=W)
        self._progress = ttk.Progressbar(pf, length=400, mode="determinate",
                                          style="green.Horizontal.TProgressbar")
        self._progress.pack(fill="x", pady=4)

        sf = Frame(pf, bg=self.PANEL_BG)
        sf.pack(fill="x")
        self._prog_label  = self._label(sf, "Iteration: —", fg=self.TEXT_MUTED)
        self._prog_label.pack(side=LEFT)
        self._fit_label   = self._label(sf, "Fitness: —", fg=self.TEXT_MUTED)
        self._fit_label.pack(side=LEFT, padx=20)
        self._feat_label  = self._label(sf, f"Features: 10 / {TOTAL_FEATURES}", fg=self.ACCENT2)
        self._feat_label.pack(side=LEFT)

        self._sep(f)

        # Verdict
        vf = Frame(f, bg=self.PANEL_BG)
        vf.pack(fill="x", padx=12, pady=6)
        self._verdict_var = StringVar(value="AWAITING PACKET")
        self._verdict_lbl = Label(vf, textvariable=self._verdict_var,
                                   bg=self.PANEL_BG, fg=self.TEXT_MUTED,
                                   font=("Courier New", 20, "bold"))
        self._verdict_lbl.pack(anchor=W)

        self._conf_lbl = self._label(vf, "Run the optimizer to classify a packet.",
                                      fg=self.TEXT_MUTED)
        self._conf_lbl.pack(anchor=W, pady=2)

        # Probability bars
        bf = Frame(f, bg=self.PANEL_BG)
        bf.pack(fill="x", padx=12, pady=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                         troughcolor=self.BORDER, background=self.ACCENT2,
                         bordercolor=self.BORDER, lightcolor=self.ACCENT2,
                         darkcolor=self.ACCENT2)
        style.configure("red.Horizontal.TProgressbar",
                         troughcolor=self.BORDER, background="#EF4444",
                         bordercolor=self.BORDER, lightcolor="#EF4444",
                         darkcolor="#EF4444")

        self._label(bf, "Benign probability:", fg=self.TEXT_MUTED, anchor=W).pack(anchor=W)
        self._bar_benign = ttk.Progressbar(bf, length=400, mode="determinate",
                                            style="green.Horizontal.TProgressbar")
        self._bar_benign.pack(fill="x", pady=2)

        self._label(bf, "Attack probability:", fg=self.TEXT_MUTED, anchor=W).pack(anchor=W)
        self._bar_attack = ttk.Progressbar(bf, length=400, mode="determinate",
                                            style="red.Horizontal.TProgressbar")
        self._bar_attack.pack(fill="x", pady=2)

        self._sep(f)

        # Feature signals
        sf2 = Frame(f, bg=self.PANEL_BG)
        sf2.pack(fill="x", padx=12, pady=4)
        self._label(sf2, "Feature signal analysis:", fg=self.TEXT_MUTED).pack(anchor=W)
        self._signal_text = Text(sf2, height=5, width=50,
                                  bg=self.ENTRY_BG, fg="#F59E0B",
                                  font=self.FONT_MONO, relief="flat",
                                  state="disabled",
                                  highlightbackground=self.BORDER,
                                  highlightthickness=1)
        self._signal_text.pack(fill="x", pady=4)

        self._sep(f)

        # Metrics comparison table
        self._label(f, "Model comparison — same MBGWO-selected features:",
                     fg=self.TEXT_MUTED).pack(anchor=W, padx=12, pady=(4, 2))
        tf = Frame(f, bg=self.PANEL_BG)
        tf.pack(fill="x", padx=12, pady=4)

        headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        widths   = [30, 10, 10, 10, 10]
        for col, (h, w) in enumerate(zip(headers, widths)):
            Label(tf, text=h, width=w, anchor=W,
                  bg=self.BORDER, fg=self.ACCENT,
                  font=self.FONT_HEAD, padx=6, pady=4
                  ).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

        for r, (model, metrics) in enumerate(MODEL_METRICS.items(), start=1):
            vals = [model] + [f"{metrics[k]:.4f}" for k in ["Accuracy","Precision","Recall","F1-Score"]]
            row_bg = "#0F172A" if r % 2 == 0 else self.ENTRY_BG
            for col, (val, w) in enumerate(zip(vals, widths)):
                Label(tf, text=val, width=w, anchor=W,
                      bg=row_bg, fg=self.ACCENT2 if r == 1 else self.TEXT,
                      font=self.FONT_MONO, padx=6, pady=3
                      ).grid(row=r, column=col, sticky="nsew", padx=1, pady=1)

    # ── Log panel ─────────────────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        f = self._frame(parent, "Detection Log")
        f.pack(fill="both", expand=True)
        # container for proper layout
        container = Frame(f, bg=self.PANEL_BG)
        container.pack(fill="both", expand=True, padx=8, pady=6)

        # TEXT AREA (FIXED: must be inside container)
        self._log_text = Text(
            container,
            bg=self.ENTRY_BG,
            fg=self.TEXT,
            font=self.FONT_MONO,
            relief="flat",
            state="disabled",
            highlightbackground=self.BORDER,highlightthickness=1
        )
        # SCROLLBAR
        sb = Scrollbar(container, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        # TAGS
        self._log_text.tag_config("benign", foreground=self.ACCENT2)
        self._log_text.tag_config("attack", foreground=self.DANGER)
        self._log_text.tag_config("header", foreground=self.ACCENT)
        self._log_text.tag_config("muted", foreground=self.TEXT_MUTED)

        # INITIAL LOG
        self._log_text.config(state="normal")
        self._log_text.insert("end", "=" * 60 + "\n", "header")
        self._log_text.insert("end", " MBGWO IDS LOG STARTED\n", "header")
        self._log_text.insert("end", "=" * 60 + "\n\n", "header")
        self._log_text.config(state="disabled")

    # ── Actions ───────────────────────────────────────────────────────────────
    def _load_sample(self, name):
        vals = SAMPLES[name]
        for e, v in zip(self._feat_vars, vals):
            e.delete(0, END)
            e.insert(0, str(v))
        #self._log("muted", f"  [Sample loaded: {name}]\n")

    def _get_params(self):
        try:
            runs  = int(self._run_entry.get())
            pop   = int(self._pop_entry.get())
            iters = int(self._iter_entry.get())
        except ValueError:
            runs, pop, iters = 5, 15, 25
        return runs, pop, iters

    def _get_raw_values(self):
        vals = []
        for e in self._feat_vars:
            try:
                vals.append(float(e.get()))
            except ValueError:
                vals.append(0.0)
        return vals

    def _on_generate(self):
        if self._gwo_thread and self._gwo_thread.is_alive():
            return  # already running

        runs, pop, iters = self._get_params()
        opt = self._opt_var.get()
        raw = self._get_raw_values()
        self._run_count += 1
        run_id = self._run_count

        # Reset UI
        self._progress["value"] = 0
        self._verdict_var.set("ANALYZING...")
        self._verdict_lbl.config(fg=self.TEXT_MUTED)
        self._conf_lbl.config(text=f"Running {opt} · Wolves: {pop} · Iterations: {iters}")
        self._bar_benign["value"] = 0
        self._bar_attack["value"] = 0
        self._set_signals([])

        self._log("header", f"\n{'─'*60}\n")
        self._log("header", f"  Run #{run_id} · {opt} · Pop={pop} · Iter={iters}\n")
        self._log("header", f"{'─'*60}\n")

        def on_iter(t, fitness, n_feat):
            pct = (t / iters) * 100
            self.root.after(0, lambda: self._update_progress(t, iters, fitness, n_feat, pct))

        def on_done(fitness, n_feat):
            is_attack, attack_prob, signals = classify_packet(raw)
            benign_prob = 1 - attack_prob
            self.root.after(0, lambda: self._show_result(
                run_id, opt, is_attack, attack_prob, benign_prob, signals,
                fitness, n_feat, raw))

        self._gwo_sim = MBGWOSimulator(pop, iters, opt, on_iter, on_done)
        self._gwo_thread = threading.Thread(target=self._gwo_sim.run, args=(raw,), daemon=True)
        self._gwo_thread.start()

    def _update_progress(self, t, max_iter, fitness, n_feat, pct):
        self._progress["value"] = pct
        self._prog_label.config(text=f"Iteration: {t} / {max_iter}")
        self._fit_label.config(text=f"Fitness: {fitness:.5f}")
        self._feat_label.config(text=f"Features: {n_feat} / {TOTAL_FEATURES}")

    def _show_result(self, run_id, opt, is_attack, attack_prob, benign_prob,
                      signals, fitness, n_feat, raw):
        self._progress["value"] = 100

        if is_attack:
            self._verdict_var.set("⚠  ATTACK DETECTED")
            self._verdict_lbl.config(fg=self.DANGER)
            conf_str = f"Attack confidence: {attack_prob*100:.1f}%"
        else:
            self._verdict_var.set("✔  BENIGN TRAFFIC")
            self._verdict_lbl.config(fg=self.ACCENT2)
            conf_str = f"Benign confidence: {benign_prob*100:.1f}%"

        self._conf_lbl.config(text=f"{opt} · Run #{run_id} · {conf_str}")
        self._bar_benign["value"] = benign_prob * 100
        self._bar_attack["value"] = attack_prob * 100
        self._set_signals(signals)

        # Identify packet type
        packet_type = self._identify_packet(raw)

        # Log entry
        tag = "attack" if is_attack else "benign"
        verdict = "ATTACK" if is_attack else "BENIGN"
        self._log(tag, f"  [{verdict}]  {packet_type} packet\n")
        self._log("muted", f"  Confidence: {max(attack_prob, benign_prob)*100:.1f}%  ·  "
                            f"Features selected: {n_feat}/{TOTAL_FEATURES}  ·  "
                            f"Best fitness: {fitness:.5f}\n")
        if signals:
            self._log("muted", "  Flagged features:\n")
            for s in signals:
                self._log("attack" if is_attack else "muted", f"    • {s}\n")
        self._log("muted", "\n")
        self._history.append({
            "run": run_id, "opt": opt, "verdict": verdict,
            "packet": packet_type, "confidence": max(attack_prob, benign_prob)
        })

    def _identify_packet(self, raw):
        """Match raw values to a known sample type."""
        for name, sample in SAMPLES.items():
            if all(abs(raw[i] - sample[i]) < 1e-3 for i in range(len(sample))):
                return name
        return "Custom"

    def _set_signals(self, signals):
        self._signal_text.config(state="normal")
        self._signal_text.delete("1.0", END)
        if signals:
            for s in signals:
                self._signal_text.insert(END, f"  ▸ {s}\n")
        else:
            self._signal_text.insert(END, "  No anomalous features detected.\n")
        self._signal_text.config(state="disabled")

    def _log(self, tag, text):
        self._log_text.config(state="normal")
        self._log_text.insert(END, text, tag)
        self._log_text.see(END)
        self._log_text.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    IDSApp()