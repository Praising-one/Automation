import re
import colorsys
import subprocess
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    FEATURE_C_MPL_AVAILABLE = True
except Exception:
    FigureCanvasTkAgg = None
    Figure = None
    FEATURE_C_MPL_AVAILABLE = False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Work Automation Tool")
        self.geometry("980x620")
        self.minsize(900, 560)

        self.shared_state = {
            "last_file_path": "",
            "feature_a_result_df": None,
            "feature_c_ball_map_df": None,
        }

        self._build_style()

        self.container = ttk.Frame(self, padding=12)
        self.container.pack(fill="both", expand=True)

        self.current_page = None
        self.show_home()

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Malgun Gothic", 16, "bold"))
        style.configure("SubTitle.TLabel", font=("Malgun Gothic", 10))
        style.configure("Menu.TButton", font=("Malgun Gothic", 11), padding=10)

    def _switch_page(self, page_cls, **kwargs):
        if self.current_page is not None:
            self.current_page.destroy()
        self.current_page = page_cls(self.container, app=self, **kwargs)
        self.current_page.pack(fill="both", expand=True)

    def show_home(self):
        self._switch_page(HomePage)

    def show_feature(self, feature_key):
        self._switch_page(FeaturePage, feature_key=feature_key)


class HomePage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.features = [
            {
                "key": "feature_a",
                "name": "Feature A - CSV/TXT Net Pair View",
                "desc": "Load tab-separated data and compare IN/OUT rows side by side.",
            },
            {
                "key": "feature_b",
                "name": "Feature B - TXT Analysis",
                "desc": "Placeholder.",
            },
            {
                "key": "feature_c",
                "name": "Feature C - Ball Map Visualizer",
                "desc": "Load ball map and highlight pasted bad-net list.",
            },
        ]

        self._build_ui()

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(header, text="Work Automation Tool", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Select a feature to run.",
            style="SubTitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        for item in self.features:
            card = ttk.Frame(body, padding=10)
            card.pack(fill="x", pady=6)

            ttk.Button(
                card,
                text=item["name"],
                style="Menu.TButton",
                command=lambda key=item["key"]: self.app.show_feature(key),
            ).pack(side="left")

            ttk.Label(card, text=item["desc"]).pack(side="left", padx=12)


class FeaturePage(ttk.Frame):
    FEATURE_C_PAGE_SIZE = 20
    FEATURE_C_TABLE_PREVIEW_ROWS = 1000

    FEATURE_META = {
        "feature_a": "Feature A - CSV/TXT Net Pair View",
        "feature_b": "Feature B - TXT Analysis",
        "feature_c": "Feature C - Ball Map Visualizer",
    }

    def __init__(self, parent, app, feature_key):
        super().__init__(parent)
        self.app = app
        self.feature_key = feature_key

        self.input_file_var = tk.StringVar(value=self.app.shared_state.get("last_file_path", ""))
        self.log_window = None
        self.log_text = None
        self.log_buffer = []
        self.result_tree = None
        self.feature_c_paste_text = None
        self.feature_c_plot_host = None
        self.feature_c_legend_host = None
        self.feature_c_mpl_figure = None
        self.feature_c_mpl_canvas = None
        self.feature_c_mpl_axes = None
        self.feature_c_mpl_legend_axes = None
        self.feature_c_mpl_legend_figure = None
        self.feature_c_mpl_legend_canvas = None
        self.feature_c_df = pd.DataFrame()
        self.feature_c_net_col = ""
        self.feature_c_x_col = ""
        self.feature_c_y_col = ""
        self.feature_c_points = []
        self.feature_c_item_to_net = {}
        self.feature_c_active_nets = set()
        self.feature_c_display_net_by_norm = {}
        self.feature_c_color_by_net = {}
        self.feature_c_matched_nets = []
        self.feature_c_highlight_pages = []
        self.feature_c_page_index = 0
        self.feature_c_page_label_var = tk.StringVar(value="Page 0/0")
        self.feature_c_prev_btn = None
        self.feature_c_next_btn = None
        self.feature_c_toggle_loaded_btn = None
        self.feature_c_loaded_frame = None
        self.feature_c_loaded_visible = False

        self._build_ui()

    def _build_ui(self):
        title = self.FEATURE_META.get(self.feature_key, "Feature")

        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 10))

        ttk.Button(top, text="Home", command=self.app.show_home).pack(side="left")
        ttk.Label(top, text=title, style="Title.TLabel").pack(side="left", padx=12)
        ttk.Button(top, text="View Logs", command=self._open_log_window).pack(side="right")

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)
        main.rowconfigure(1, weight=1)
        if self.feature_key == "feature_c":
            main.rowconfigure(2, weight=0)
        main.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(main, text="Input", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 8))
        input_frame.columnconfigure(0, weight=1)

        file_row = ttk.Frame(input_frame)
        file_row.pack(fill="x", pady=(0, 8))
        ttk.Label(file_row, text="Input file:").pack(side="left")
        ttk.Entry(file_row, textvariable=self.input_file_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(file_row, text="Browse", command=self._choose_file).pack(side="left")

        if self.feature_key == "feature_c":
            paste_row = ttk.Frame(input_frame)
            paste_row.pack(fill="x", pady=(0, 8))
            ttk.Label(paste_row, text="Bad-net paste:").pack(anchor="w")
            self.feature_c_paste_text = tk.Text(paste_row, height=4, wrap="none")
            self.feature_c_paste_text.pack(fill="x", expand=True, pady=(4, 0))

            actions_row = ttk.Frame(input_frame)
            actions_row.pack(fill="x", pady=(0, 2))
            ttk.Button(actions_row, text="Apply Highlight", command=self._apply_feature_c_highlight).pack(side="left")
            ttk.Button(actions_row, text="Clear Highlight", command=self._clear_feature_c_highlight).pack(side="left", padx=(6, 0))
            ttk.Button(actions_row, text="Copy Plot", command=self._copy_feature_c_plot_image).pack(side="left", padx=(12, 0))
            ttk.Button(actions_row, text="Copy Legend", command=self._copy_feature_c_legend_image).pack(side="left", padx=(6, 0))
            self.feature_c_toggle_loaded_btn = ttk.Button(
                actions_row,
                text="Show Loaded Data",
                command=self._toggle_feature_c_loaded_data,
            )
            self.feature_c_toggle_loaded_btn.pack(side="left", padx=(6, 0))
            self.feature_c_prev_btn = ttk.Button(actions_row, text="Prev", command=self._feature_c_prev_page)
            self.feature_c_prev_btn.pack(side="right")
            self.feature_c_next_btn = ttk.Button(actions_row, text="Next", command=self._feature_c_next_page)
            self.feature_c_next_btn.pack(side="right", padx=(6, 0))
            ttk.Label(actions_row, textvariable=self.feature_c_page_label_var).pack(side="right", padx=(0, 10))

        ttk.Button(input_frame, text="Run", command=self.run_feature).pack(anchor="e", pady=(8, 0))

        if self.feature_key == "feature_c":
            plot_frame = ttk.LabelFrame(main, text="Ball Map", padding=10)
            plot_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=(0, 8))
            plot_frame.rowconfigure(0, weight=1)
            plot_frame.rowconfigure(1, weight=0)
            plot_frame.columnconfigure(0, weight=1)
            self.feature_c_plot_host = ttk.Frame(plot_frame)
            self.feature_c_plot_host.grid(row=0, column=0, sticky="nsew")
            self.feature_c_legend_host = ttk.LabelFrame(plot_frame, text="Legend (Current Page)", padding=6)
            self.feature_c_legend_host.grid(row=1, column=0, sticky="ew", pady=(8, 0))

            if FEATURE_C_MPL_AVAILABLE:
                fig = Figure(figsize=(10.2, 6.2), dpi=100)
                self.feature_c_mpl_axes = fig.add_subplot(111)
                self.feature_c_mpl_figure = fig
                self.feature_c_mpl_canvas = FigureCanvasTkAgg(fig, master=self.feature_c_plot_host)
                self.feature_c_mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

                legend_fig = Figure(figsize=(10.2, 1.8), dpi=100)
                self.feature_c_mpl_legend_axes = legend_fig.add_subplot(111)
                self.feature_c_mpl_legend_axes.axis("off")
                self.feature_c_mpl_legend_figure = legend_fig
                self.feature_c_mpl_legend_canvas = FigureCanvasTkAgg(legend_fig, master=self.feature_c_legend_host)
                self.feature_c_mpl_legend_canvas.get_tk_widget().pack(fill="x", expand=True)
            else:
                ttk.Label(
                    self.feature_c_plot_host,
                    text="matplotlib is not available. Install with:\npython -m pip install matplotlib",
                ).pack(fill="both", expand=True, padx=8, pady=8)
                ttk.Label(
                    self.feature_c_legend_host,
                    text="Legend requires matplotlib.",
                ).pack(fill="x", expand=True, padx=8, pady=4)

            self.feature_c_loaded_frame = ttk.LabelFrame(main, text="Loaded Data", padding=10)
            self.feature_c_loaded_frame.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
            self._build_result_tree(self.feature_c_loaded_frame)
            self.feature_c_loaded_frame.grid_remove()
        else:
            result_frame = ttk.LabelFrame(main, text="Result DataFrame", padding=10)
            result_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
            self._build_result_tree(result_frame)

        self._append_log(f"[Info] Feature logic is ready for {title}.")
        if self.feature_key == "feature_c":
            self._update_feature_c_page_controls()
            self._draw_feature_c_legend()

    def _build_result_tree(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.result_tree = ttk.Treeview(parent, show="headings")
        self.result_tree.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.result_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=self.result_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.result_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _open_log_window(self):
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            self.log_window.focus_force()
            return

        self.log_window = tk.Toplevel(self)
        self.log_window.title("Logs")
        self.log_window.geometry("900x360")

        wrapper = ttk.Frame(self.log_window, padding=10)
        wrapper.pack(fill="both", expand=True)
        wrapper.rowconfigure(0, weight=1)
        wrapper.columnconfigure(0, weight=1)

        self.log_text = tk.Text(wrapper, height=16, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(wrapper, orient="vertical", command=self.log_text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(wrapper, orient="horizontal", command=self.log_text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        if self.log_buffer:
            self.log_text.insert("end", "\n".join(self.log_buffer) + "\n")
            self.log_text.see("end")

    def _show_result_dataframe(self, df: pd.DataFrame):
        self.result_tree.delete(*self.result_tree.get_children())
        show_df = df
        if self.feature_key == "feature_c" and len(df) > self.FEATURE_C_TABLE_PREVIEW_ROWS:
            show_df = df.head(self.FEATURE_C_TABLE_PREVIEW_ROWS).copy()
            self._append_log(
                f"[Feature C] Loaded Data preview: showing first {len(show_df):,} of {len(df):,} rows."
            )

        columns = list(show_df.columns)
        self.result_tree["columns"] = columns
        if not columns:
            return

        for col in show_df.columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=140, anchor="w", stretch=True)

        for row in show_df.astype(str).itertuples(index=False, name=None):
            self.result_tree.insert("", "end", values=row)

    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("Data files", "*.csv;*.txt"), ("All files", "*.*")],
        )
        if path:
            self.input_file_var.set(path)
            self.app.shared_state["last_file_path"] = path
            self._append_log(f"[Selected] {path}")

    def run_feature(self):
        if self.feature_key == "feature_a":
            self._run_feature_a()
        elif self.feature_key == "feature_b":
            self._run_feature_b()
        elif self.feature_key == "feature_c":
            self._run_feature_c()
        else:
            self._append_log("[Error] Unknown feature key.")

    def _run_feature_a(self):
        path_text = self.input_file_var.get().strip()
        if not path_text:
            messagebox.showwarning("Input required", "Select a CSV/TXT file first.")
            return

        file_path = Path(path_text)
        if not file_path.exists():
            messagebox.showerror("File not found", f"Input file does not exist:\n{file_path}")
            return

        try:
            df = self._read_table_file(file_path)
        except Exception as exc:
            self._append_log(f"[Error] Failed to read file: {exc}")
            messagebox.showerror("Read failed", str(exc))
            return

        net_col = self._find_net_column(df)
        if not net_col:
            msg = "Required column not found. Expected one of: 'Net Name' or 'Net'."
            self._append_log(f"[Error] {msg}")
            messagebox.showerror("Column error", msg)
            return

        try:
            result_df = self._build_in_out_view(df, net_col)
        except Exception as exc:
            self._append_log(f"[Error] Failed to build IN/OUT view: {exc}")
            messagebox.showerror("Feature A failed", str(exc))
            return

        if result_df.empty:
            msg = "No IN/OUT rows matched after parsing net names."
            self._append_log(f"[Info] {msg}")
            self._show_result_dataframe(pd.DataFrame())
            messagebox.showinfo("Feature A", msg)
            return

        self.app.shared_state["feature_a_result_df"] = result_df
        self._show_result_dataframe(result_df)

        output_path = file_path.with_name(f"{file_path.stem}_in_out_view.csv")
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        self._append_log(f"[Feature A] Input rows: {len(df):,}")
        self._append_log(f"[Feature A] Matched rows: {len(result_df):,}")
        self._append_log(f"[Feature A] Saved result: {output_path}")

        messagebox.showinfo("Feature A complete", f"Result saved to:\n{output_path}")

    def _read_table_file(self, file_path: Path) -> pd.DataFrame:
        last_error = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
            try:
                return pd.read_csv(file_path, sep="\t", encoding=enc, dtype=str, keep_default_na=False)
            except Exception as exc:  # noqa: PERF203
                last_error = exc
        raise ValueError(f"Could not read file with tab separator. Last error: {last_error}")

    def _read_table_file_auto(self, file_path: Path) -> pd.DataFrame:
        separators = [None, ",", "\t", ";", "|"]
        last_error = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
            for sep in separators:
                try:
                    kwargs = {
                        "encoding": enc,
                        "dtype": str,
                        "keep_default_na": False,
                    }
                    if sep is None:
                        kwargs["sep"] = None
                        kwargs["engine"] = "python"
                    else:
                        kwargs["sep"] = sep
                    df = pd.read_csv(file_path, **kwargs)
                    if not df.empty and len(df.columns) >= 2:
                        return df
                except Exception as exc:  # noqa: PERF203
                    last_error = exc
        raise ValueError(f"Could not read file as CSV/TXT. Last error: {last_error}")

    def _find_net_column(self, df: pd.DataFrame) -> str:
        for name in ("Net Name", "Net"):
            if name in df.columns:
                return name
        return ""

    def _get_column_map(self, df: pd.DataFrame):
        col_map = {}
        for col in df.columns:
            norm = re.sub(r"\s+", " ", str(col).strip()).casefold()
            col_map[norm] = col
        return col_map

    def _find_feature_c_columns(self, df: pd.DataFrame):
        col_map = self._get_column_map(df)

        net_col = ""
        for candidate in ("Net", "Net Name"):
            found = col_map.get(candidate.casefold())
            if found:
                net_col = found
                break

        xy_pair = ("", "")
        for x_candidate, y_candidate in (("XLocation", "YLocation"), ("X Coord", "Y Coord")):
            x_found = col_map.get(x_candidate.casefold())
            y_found = col_map.get(y_candidate.casefold())
            if x_found and y_found:
                xy_pair = (x_found, y_found)
                break

        if not net_col or not xy_pair[0] or not xy_pair[1]:
            return "", "", ""
        return net_col, xy_pair[0], xy_pair[1]

    def _parse_net_name(self, raw_name: str):
        text = str(raw_name).strip()
        if not text:
            return None, None, None

        parts = re.split(r"([_\-\s]+)", text)
        direction = None
        kept_parts = []

        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                low = part.lower()
                if low == "in" and direction is None:
                    direction = "IN"
                    continue
                if low == "out" and direction is None:
                    direction = "OUT"
                    continue
            kept_parts.append(part)

        if direction is None:
            return None, None, None

        display_key = "".join(kept_parts).strip("_- ")
        display_key = re.sub(r"[_\-\s]+", "_", display_key).strip("_")
        if not display_key:
            return None, None, None

        norm_key = display_key.casefold()
        return direction, norm_key, display_key

    def _build_in_out_view(self, df: pd.DataFrame, net_col: str) -> pd.DataFrame:
        work = df.copy()
        parsed = work[net_col].map(self._parse_net_name)
        parsed_df = pd.DataFrame(parsed.tolist(), columns=["_dir", "_pair_key_norm", "_pair_key"])
        work = pd.concat([work, parsed_df], axis=1)
        work = work[work["_dir"].isin(["IN", "OUT"])].copy()

        if work.empty:
            return pd.DataFrame()

        in_df = work[work["_dir"] == "IN"].copy()
        out_df = work[work["_dir"] == "OUT"].copy()

        if in_df.empty and out_df.empty:
            return pd.DataFrame()

        in_counts = in_df.groupby("_pair_key_norm").size().to_dict()
        out_counts = out_df.groupby("_pair_key_norm").size().to_dict()

        if not in_df.empty:
            in_df["_seq"] = in_df.groupby("_pair_key_norm").cumcount() + 1
        if not out_df.empty:
            out_df["_seq"] = out_df.groupby("_pair_key_norm").cumcount() + 1

        original_cols = list(df.columns)
        in_rename = {c: f"IN__{c}" for c in original_cols}
        out_rename = {c: f"OUT__{c}" for c in original_cols}

        in_keep = ["_pair_key_norm", "_pair_key", "_seq"] + original_cols
        out_keep = ["_pair_key_norm", "_pair_key", "_seq"] + original_cols

        in_view = in_df[in_keep].rename(columns=in_rename)
        out_view = out_df[out_keep].rename(columns=out_rename)

        merged = in_view.merge(
            out_view,
            on=["_pair_key_norm", "_seq"],
            how="outer",
            suffixes=("_IN", "_OUT"),
        )

        merged["PAIR_KEY"] = merged["_pair_key_IN"].fillna(merged["_pair_key_OUT"])
        merged["ROW_INDEX"] = merged["_seq"].astype("Int64")
        merged["IN_COUNT"] = merged["_pair_key_norm"].map(in_counts).fillna(0).astype(int)
        merged["OUT_COUNT"] = merged["_pair_key_norm"].map(out_counts).fillna(0).astype(int)

        ordered = ["PAIR_KEY", "ROW_INDEX", "IN_COUNT", "OUT_COUNT"]
        ordered += [f"IN__{c}" for c in original_cols]
        ordered += [f"OUT__{c}" for c in original_cols]

        result = merged[ordered].sort_values(by=["PAIR_KEY", "ROW_INDEX"], kind="stable")
        result = result.reset_index(drop=True)
        return result

    def _run_feature_b(self):
        self._append_log("[Feature B] Placeholder.")
        messagebox.showinfo("Info", "Feature B logic is not implemented yet.")

    def _run_feature_c(self):
        path_text = self.input_file_var.get().strip()
        if not path_text:
            messagebox.showwarning("Input required", "Select a CSV/TXT file first.")
            return

        file_path = Path(path_text)
        if not file_path.exists():
            messagebox.showerror("File not found", f"Input file does not exist:\n{file_path}")
            return

        try:
            df = self._read_table_file_auto(file_path)
        except Exception as exc:
            self._append_log(f"[Error] Failed to read Feature C input: {exc}")
            messagebox.showerror("Read failed", str(exc))
            return

        net_col, x_col, y_col = self._find_feature_c_columns(df)
        if not net_col or not x_col or not y_col:
            msg = (
                "Required columns not found.\n"
                "Net: 'Net' or 'Net Name'\n"
                "XY: ('XLocation','YLocation') or ('X Coord','Y Coord')"
            )
            self._append_log(f"[Error] {msg}")
            messagebox.showerror("Column error", msg)
            return

        points = self._build_feature_c_points(df, net_col, x_col, y_col)
        if not points:
            msg = f"No valid numeric coordinates found in columns: {x_col}, {y_col}"
            self._append_log(f"[Error] {msg}")
            messagebox.showerror("Data error", msg)
            return

        self.feature_c_df = df.copy()
        self.feature_c_net_col = net_col
        self.feature_c_x_col = x_col
        self.feature_c_y_col = y_col
        self.feature_c_points = points
        self.feature_c_active_nets.clear()
        self.feature_c_color_by_net = {}
        self.feature_c_matched_nets = []
        self.feature_c_highlight_pages = []
        self.feature_c_page_index = 0
        self.feature_c_display_net_by_norm = {}
        for point in self.feature_c_points:
            if point["net_norm"] and point["net_norm"] not in self.feature_c_display_net_by_norm:
                self.feature_c_display_net_by_norm[point["net_norm"]] = point["net"]
        self.app.shared_state["feature_c_ball_map_df"] = self.feature_c_df
        self._show_result_dataframe(self.feature_c_df)
        self._update_feature_c_page_controls()
        self._draw_feature_c_ball_map()

        self._append_log(f"[Feature C] Loaded rows: {len(df):,}")
        self._append_log(f"[Feature C] Plotted points: {len(points):,}")
        self._append_log(f"[Feature C] Columns: Net={net_col}, X={x_col}, Y={y_col}")

    def _to_float(self, value):
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _normalize_net(self, net_name: str):
        return str(net_name).strip().casefold()

    def _build_feature_c_points(self, df: pd.DataFrame, net_col: str, x_col: str, y_col: str):
        points = []
        for idx, row in df.iterrows():
            x = self._to_float(row.get(x_col, ""))
            y = self._to_float(row.get(y_col, ""))
            if x is None or y is None:
                continue
            net = str(row.get(net_col, "")).strip()
            points.append(
                {
                    "idx": idx,
                    "x": x,
                    "y": y,
                    "net": net,
                    "net_norm": self._normalize_net(net),
                }
            )
        return points

    def _paginate_items(self, items, page_size):
        if page_size <= 0:
            return [list(items)] if items else []
        return [list(items[idx : idx + page_size]) for idx in range(0, len(items), page_size)]

    def _build_feature_c_color_map(self, page_nets):
        palette = [
            "#e53935",
            "#1e88e5",
            "#43a047",
            "#fb8c00",
            "#8e24aa",
            "#00897b",
            "#f4511e",
            "#3949ab",
            "#7cb342",
            "#6d4c41",
        ]
        color_map = {}
        for idx, net_norm in enumerate(page_nets):
            if idx < len(palette):
                color_map[net_norm] = palette[idx]
            else:
                hue = (idx * 0.61803398875) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
                color_map[net_norm] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        return color_map

    def _prepare_feature_c_highlight_nets(self, pasted_nets, available_nets):
        seen = set()
        ordered_requested = []
        display_by_norm = {}
        for raw in pasted_nets:
            value = str(raw).strip()
            if not value:
                continue
            net_norm = self._normalize_net(value)
            if net_norm not in seen:
                seen.add(net_norm)
                ordered_requested.append(net_norm)
                display_by_norm[net_norm] = value

        matched = [net_norm for net_norm in ordered_requested if net_norm in available_nets]
        return matched, display_by_norm

    def _update_feature_c_page_controls(self):
        total_pages = len(self.feature_c_highlight_pages)
        if total_pages == 0:
            self.feature_c_page_label_var.set("Page 0/0")
            if self.feature_c_prev_btn is not None:
                self.feature_c_prev_btn.configure(state="disabled")
            if self.feature_c_next_btn is not None:
                self.feature_c_next_btn.configure(state="disabled")
            return

        self.feature_c_page_index = max(0, min(self.feature_c_page_index, total_pages - 1))
        self.feature_c_page_label_var.set(f"Page {self.feature_c_page_index + 1}/{total_pages}")

        if self.feature_c_prev_btn is not None:
            prev_state = "normal" if self.feature_c_page_index > 0 else "disabled"
            self.feature_c_prev_btn.configure(state=prev_state)
        if self.feature_c_next_btn is not None:
            next_state = "normal" if self.feature_c_page_index < total_pages - 1 else "disabled"
            self.feature_c_next_btn.configure(state=next_state)

    def _feature_c_prev_page(self):
        if self.feature_c_page_index <= 0:
            return
        self.feature_c_page_index -= 1
        page_nets = self.feature_c_highlight_pages[self.feature_c_page_index]
        self.feature_c_color_by_net = self._build_feature_c_color_map(page_nets)
        self._update_feature_c_page_controls()
        self._draw_feature_c_ball_map()

    def _feature_c_next_page(self):
        if self.feature_c_page_index >= len(self.feature_c_highlight_pages) - 1:
            return
        self.feature_c_page_index += 1
        page_nets = self.feature_c_highlight_pages[self.feature_c_page_index]
        self.feature_c_color_by_net = self._build_feature_c_color_map(page_nets)
        self._update_feature_c_page_controls()
        self._draw_feature_c_ball_map()

    def _on_feature_c_canvas_resize(self, _event):
        if self.feature_key == "feature_c" and self.feature_c_points:
            self._draw_feature_c_ball_map()

    def _toggle_feature_c_loaded_data(self):
        if self.feature_key != "feature_c" or self.feature_c_loaded_frame is None:
            return
        self.feature_c_loaded_visible = not self.feature_c_loaded_visible
        if self.feature_c_loaded_visible:
            self.feature_c_loaded_frame.grid()
            if self.feature_c_toggle_loaded_btn is not None:
                self.feature_c_toggle_loaded_btn.configure(text="Hide Loaded Data")
        else:
            self.feature_c_loaded_frame.grid_remove()
            if self.feature_c_toggle_loaded_btn is not None:
                self.feature_c_toggle_loaded_btn.configure(text="Show Loaded Data")

    def _draw_feature_c_ball_map(self):
        if not FEATURE_C_MPL_AVAILABLE:
            return
        if self.feature_c_mpl_axes is None or self.feature_c_mpl_canvas is None:
            return

        ax = self.feature_c_mpl_axes
        ax.clear()

        if not self.feature_c_points:
            ax.text(0.02, 0.96, "Run to load a ball map file.", transform=ax.transAxes, va="top", color="#555555")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("white")
            self._draw_feature_c_legend()
            self.feature_c_mpl_canvas.draw_idle()
            return

        xs = [p["x"] for p in self.feature_c_points]
        ys = [p["y"] for p in self.feature_c_points]
        color_values = []
        for point in self.feature_c_points:
            if point["net_norm"] in self.feature_c_color_by_net:
                color_values.append(self.feature_c_color_by_net[point["net_norm"]])
            elif point["net_norm"] in self.feature_c_active_nets:
                color_values.append("#c7c7c7")
            else:
                color_values.append("#8e8e8e")

        ax.scatter(xs, ys, c=color_values, s=7, marker="o", linewidths=0, rasterized=True)
        x_min, x_max, y_min, y_max = self._get_feature_c_axis_limits(xs, ys)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
        ax.set_facecolor("white")
        ax.set_xlabel(self.feature_c_x_col or "X")
        ax.set_ylabel(self.feature_c_y_col or "Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        total_pages = len(self.feature_c_highlight_pages)
        if total_pages > 0:
            page_text = f"Page {self.feature_c_page_index + 1}/{total_pages} | Nets on page: {len(self.feature_c_color_by_net)}"
        else:
            page_text = "Page 0/0 | Nets on page: 0"
        ax.set_title(
            f"Gray: normal | Light-gray: highlighted on other pages | Points: {len(self.feature_c_points):,} | {page_text}",
            fontsize=9,
        )

        self._draw_feature_c_legend()
        self.feature_c_mpl_canvas.draw_idle()

    def _draw_feature_c_legend(self):
        if not FEATURE_C_MPL_AVAILABLE:
            return
        if self.feature_c_mpl_legend_axes is None:
            return
        self._draw_feature_c_legend_on_axes(self.feature_c_mpl_legend_axes)
        if self.feature_c_mpl_legend_canvas is not None:
            self.feature_c_mpl_legend_canvas.draw_idle()

    def _extract_feature_c_paste_nets(self):
        if self.feature_c_paste_text is None:
            return []
        raw = self.feature_c_paste_text.get("1.0", "end").strip()
        if not raw:
            return []

        tokens = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            elif "," in line:
                cells = [c.strip() for c in line.split(",")]
            else:
                cells = [line]
            for cell in cells:
                if cell:
                    tokens.append(cell)
        return tokens

    def _apply_feature_c_highlight(self):
        if self.feature_key != "feature_c":
            return
        if not self.feature_c_points:
            messagebox.showwarning("Load required", "Load a ball map file with Run first.")
            return

        pasted = self._extract_feature_c_paste_nets()
        if not pasted:
            messagebox.showwarning("Input required", "Paste net names first.")
            return

        available_nets = {p["net_norm"] for p in self.feature_c_points}
        matched_nets, pasted_name_map = self._prepare_feature_c_highlight_nets(pasted, available_nets)
        self.feature_c_active_nets = set(matched_nets)
        self.feature_c_matched_nets = matched_nets
        self.feature_c_display_net_by_norm.update(pasted_name_map)
        self.feature_c_highlight_pages = self._paginate_items(self.feature_c_matched_nets, self.FEATURE_C_PAGE_SIZE)
        self.feature_c_page_index = 0
        page_nets = self.feature_c_highlight_pages[0] if self.feature_c_highlight_pages else []
        self.feature_c_color_by_net = self._build_feature_c_color_map(page_nets)
        self._update_feature_c_page_controls()
        self._draw_feature_c_ball_map()

        hit_points = sum(1 for p in self.feature_c_points if p["net_norm"] in self.feature_c_active_nets)
        requested_count = len({self._normalize_net(net) for net in pasted if str(net).strip()})
        self._append_log(f"[Feature C] Highlight requested nets: {requested_count:,}")
        self._append_log(
            f"[Feature C] Matched nets: {len(matched_nets):,} | Highlighted points: {hit_points:,} | "
            f"Pages: {len(self.feature_c_highlight_pages):,}"
        )

        if not self.feature_c_active_nets:
            messagebox.showinfo("Feature C", "No pasted net names matched loaded ball map nets.")

    def _clear_feature_c_highlight(self):
        if self.feature_key != "feature_c":
            return
        self.feature_c_active_nets.clear()
        self.feature_c_color_by_net = {}
        self.feature_c_matched_nets = []
        self.feature_c_highlight_pages = []
        self.feature_c_page_index = 0
        if self.feature_c_paste_text is not None:
            self.feature_c_paste_text.delete("1.0", "end")
        self._update_feature_c_page_controls()
        self._draw_feature_c_ball_map()
        self._append_log("[Feature C] Cleared highlights.")

    def _copy_feature_c_plot_image(self):
        if not self.feature_c_points:
            messagebox.showwarning("Copy failed", "Load ball map data first.")
            return
        if not FEATURE_C_MPL_AVAILABLE:
            messagebox.showerror("matplotlib required", "Install with:\npython -m pip install matplotlib")
            return
        temp_path = Path(tempfile.gettempdir()) / "feature_c_plot_hd.png"
        self._save_feature_c_plot_png(temp_path, dpi=300)
        self._copy_image_file_to_clipboard(temp_path, "plot")

    def _copy_feature_c_legend_image(self):
        if not FEATURE_C_MPL_AVAILABLE:
            messagebox.showerror("matplotlib required", "Install with:\npython -m pip install matplotlib")
            return
        temp_path = Path(tempfile.gettempdir()) / "feature_c_legend_hd.png"
        self._save_feature_c_legend_png(temp_path, dpi=300)
        self._copy_image_file_to_clipboard(temp_path, "legend")

    def _save_feature_c_plot_png(self, output_path: Path, dpi=300):
        fig = Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)

        if not self.feature_c_points:
            ax.text(0.02, 0.96, "Run to load a ball map file.", transform=ax.transAxes, va="top", color="#555555")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            xs = [p["x"] for p in self.feature_c_points]
            ys = [p["y"] for p in self.feature_c_points]
            color_values = []
            for point in self.feature_c_points:
                if point["net_norm"] in self.feature_c_color_by_net:
                    color_values.append(self.feature_c_color_by_net[point["net_norm"]])
                elif point["net_norm"] in self.feature_c_active_nets:
                    color_values.append("#c7c7c7")
                else:
                    color_values.append("#8e8e8e")
            ax.scatter(xs, ys, c=color_values, s=8, marker="o", linewidths=0, rasterized=True)
            x_min, x_max, y_min, y_max = self._get_feature_c_axis_limits(xs, ys)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color="#9e9e9e", linewidth=0.9, linestyle="--")
            ax.axvline(0, color="#9e9e9e", linewidth=0.9, linestyle="--")
            ax.set_xlabel(self.feature_c_x_col or "X")
            ax.set_ylabel(self.feature_c_y_col or "Y")
            ax.set_aspect("equal", adjustable="box")

        total_pages = len(self.feature_c_highlight_pages)
        if total_pages > 0:
            page_text = f"Page {self.feature_c_page_index + 1}/{total_pages} | Nets on page: {len(self.feature_c_color_by_net)}"
        else:
            page_text = "Page 0/0 | Nets on page: 0"
        ax.set_title(
            f"Gray: normal | Light-gray: highlighted on other pages | Points: {len(self.feature_c_points):,} | {page_text}",
            fontsize=10,
        )
        fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")

    def _get_feature_c_axis_limits(self, xs, ys):
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        max_abs_x = max(abs(min_x), abs(max_x), 1.0)
        max_abs_y = max(abs(min_y), abs(max_y), 1.0)

        pad_x = max_abs_x * 0.03
        pad_y = max_abs_y * 0.03
        return (-max_abs_x - pad_x, max_abs_x + pad_x, -max_abs_y - pad_y, max_abs_y + pad_y)

    def _save_feature_c_legend_png(self, output_path: Path, dpi=300):
        fig = Figure(figsize=(6, 8), dpi=100)
        ax = fig.add_subplot(111)
        self._draw_feature_c_legend_on_axes(ax)
        fig.savefig(output_path, dpi=dpi, facecolor="#fafafa", bbox_inches="tight")

    def _draw_feature_c_legend_on_axes(self, ax):
        from matplotlib.patches import Rectangle

        ax.clear()
        ax.axis("off")
        ax.set_facecolor("#fafafa")
        ax.text(0.02, 0.98, "Legend (Current Page)", transform=ax.transAxes, va="top", fontsize=9, color="#333333")
        if not self.feature_c_color_by_net:
            ax.text(0.02, 0.90, "No highlighted nets.", transform=ax.transAxes, va="top", color="#666666")
            return
        y = 0.90
        line_h = 0.042
        for net_norm, color in self.feature_c_color_by_net.items():
            display_name = self.feature_c_display_net_by_norm.get(net_norm, net_norm)
            ax.add_patch(Rectangle((0.02, y - 0.02), 0.06, 0.022, transform=ax.transAxes, color=color, clip_on=False))
            ax.text(0.10, y - 0.002, display_name, transform=ax.transAxes, va="center", fontsize=8, color="#222222")
            y -= line_h
            if y < 0.02:
                break

    def _copy_image_file_to_clipboard(self, image_path: Path, image_tag: str):
        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        ps_path = str(image_path).replace("'", "''")
        ps_script = (
            f"$path = '{ps_path}'; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$img = [System.Drawing.Image]::FromFile($path); "
            "[System.Windows.Forms.Clipboard]::SetImage($img); "
            "$img.Dispose();"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            self._append_log(f"[Feature C] Clipboard copy failed: {proc.stderr.strip()}")
            messagebox.showerror(
                "Clipboard failed",
                f"Could not copy image to clipboard.\nSaved file:\n{image_path}",
            )
            return

        self._append_log(f"[Feature C] Copied {image_tag} image to clipboard.")
        messagebox.showinfo("Copied", f"{image_tag.capitalize()} image copied to clipboard.")

    def _append_log(self, msg: str):
        self.log_buffer.append(msg)
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
