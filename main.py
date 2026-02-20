import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Work Automation Tool")
        self.geometry("980x620")
        self.minsize(900, 560)

        self.shared_state = {
            "last_file_path": "",
            "feature_a_result_df": None,
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
                "name": "Feature C - Scatter Preview",
                "desc": "Placeholder.",
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
    FEATURE_META = {
        "feature_a": "Feature A - CSV/TXT Net Pair View",
        "feature_b": "Feature B - TXT Analysis",
        "feature_c": "Feature C - Scatter Preview",
    }

    def __init__(self, parent, app, feature_key):
        super().__init__(parent)
        self.app = app
        self.feature_key = feature_key

        self.input_file_var = tk.StringVar(value=self.app.shared_state.get("last_file_path", ""))
        self.manual_input_var = tk.StringVar()
        self.log_window = None
        self.log_text = None
        self.log_buffer = []
        self.result_tree = None

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
        main.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(main, text="Input", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 8))
        input_frame.columnconfigure(0, weight=1)

        file_row = ttk.Frame(input_frame)
        file_row.pack(fill="x", pady=(0, 8))
        ttk.Label(file_row, text="Input file:").pack(side="left")
        ttk.Entry(file_row, textvariable=self.input_file_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(file_row, text="Browse", command=self._choose_file).pack(side="left")

        manual_row = ttk.Frame(input_frame)
        manual_row.pack(fill="x", pady=(0, 8))
        ttk.Label(manual_row, text="Manual input:").pack(side="left")
        ttk.Entry(manual_row, textvariable=self.manual_input_var).pack(side="left", fill="x", expand=True, padx=6)

        ttk.Button(input_frame, text="Run", command=self.run_feature).pack(anchor="e", pady=(8, 0))

        result_frame = ttk.LabelFrame(main, text="Result DataFrame", padding=10)
        result_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        self.result_tree = ttk.Treeview(result_frame, show="headings")
        self.result_tree.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(result_frame, orient="horizontal", command=self.result_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.result_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self._append_log("[Info] Feature logic is ready for Feature A.")

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
        columns = list(df.columns)
        self.result_tree["columns"] = columns
        if not columns:
            return

        for col in df.columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=140, anchor="w", stretch=True)

        for row in df.astype(str).itertuples(index=False, name=None):
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

    def _find_net_column(self, df: pd.DataFrame) -> str:
        for name in ("Net Name", "Net"):
            if name in df.columns:
                return name
        return ""

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
        self._append_log("[Feature C] Placeholder.")
        messagebox.showinfo("Info", "Feature C logic is not implemented yet.")

    def _append_log(self, msg: str):
        self.log_buffer.append(msg)
        if self.log_text is not None and self.log_text.winfo_exists():
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
