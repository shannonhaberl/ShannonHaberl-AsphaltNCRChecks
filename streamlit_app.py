# app.py
# Streamlit app: Asphalt NCR Checks (PDF + digital tables + header row finder + updated logic)
#
# Updated logic per Shannon's note:
# - Underlying layer cannot be higher than +U (max tolerance high). Therefore:
# * HIGH & THIN remains PHYSICALLY IMPOSSIBLE (would require underlying too high to "steal" thickness)
# * A "hard minimum feasible thickness" is pointwise:
# Tmin_point = D + LevelConformance - U
# If Thickness < Tmin_point => PHYSICALLY IMPOSSIBLE (CHECK SURVEY ERROR)
#
# - Underlying layer CAN be low (and may have an accepted NCR). Therefore:
# * LOW & THICK is NOT automatically a survey error.
# It is flagged as:
# "MAY BE ACCEPTABLE – underlying low NCR (thickness made up in asphalt)"
#
# - We also provide an informational flag:
# Thick > (D + LevelConformance + U)
# This means "thicker than would be possible IF underlying were within ±U".
# Not an impossibility, because underlying might be low beyond -U (accepted NCR).
#
# Install:
# pip install streamlit pandas numpy openpyxl camelot-py[cv]
#
# Run:
# streamlit run app.py

import os
import re
import tempfile
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Asphalt NCR Checks", layout="wide")

st.title("Asphalt NCR Checks")
st.caption(
    "Separates genuine construction nonconformances from physically impossible results "
    "using first-principles geometry (level vs thickness sanity checks)."
)

# ---------------------------
# Helpers: general
# ---------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-\+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def load_csv_excel(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    if name.endswith(".txt"):
        return pd.read_csv(upload, sep=None, engine="python")
    raise ValueError("Unsupported file type.")


# ---------------------------
# Helpers: PDF + header finding
# ---------------------------
HEADER_KEYWORDS = [
    "chainage", "ch", "offset", "off", "level", "conformance", "thickness", "ac", "rl", "mm"
]

def _cell_text(x) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _row_text(row: pd.Series) -> str:
    return " ".join(_cell_text(v) for v in row.values if _cell_text(v))


def _keyword_hits(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in HEADER_KEYWORDS if kw in t)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    return letters / max(len(text), 1)


def _numeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    nums = sum(1 for c in text if c.isdigit())
    return nums / max(len(text), 1)


def detect_header_row_index(df: pd.DataFrame, scan_rows: int = 25) -> Optional[int]:
    if df.empty:
        return None

    n = min(scan_rows, len(df))
    best_i = None
    best_score = -1.0

    for i in range(n):
        text = _row_text(df.iloc[i])
        if not text:
            continue

        hits = _keyword_hits(text)
        a = _alpha_ratio(text)
        num = _numeric_ratio(text)

        score = hits * 10.0 + a * 2.0 - num * 2.0

        if hits == 0 and num > 0.35 and a < 0.15:
            score -= 3.0

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None

    best_text = _row_text(df.iloc[best_i])
    if _keyword_hits(best_text) == 0:
        return None

    return best_i


def apply_header_row(df: pd.DataFrame, header_idx: int) -> pd.DataFrame:
    df = df.copy()
    header = [_cell_text(v) for v in df.iloc[header_idx].values]

    header_clean = []
    for j, h in enumerate(header):
        h2 = re.sub(r"\s+", " ", h).strip()
        if not h2:
            h2 = f"col_{j}"
        header_clean.append(h2)

    seen: Dict[str, int] = {}
    unique = []
    for h in header_clean:
        if h not in seen:
            seen[h] = 1
            unique.append(h)
        else:
            seen[h] += 1
            unique.append(f"{h}_{seen[h]}")

    out = df.iloc[header_idx + 1 :].copy()
    out.columns = unique
    out = out.reset_index(drop=True)
    return out


def _parse_pages(pages: str, num_pages: Optional[int]) -> List[int]:
    if not pages or pages.strip().lower() == "all":
        return list(range(1, num_pages + 1 if num_pages else 1))

    parts = [p.strip() for p in pages.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except Exception:
                continue
            out.extend(list(range(a_i, b_i + 1)))
        else:
            try:
                out.append(int(p))
            except Exception:
                continue
    return sorted(set(out))


def extract_tables_from_pdf(
    pdf_bytes: bytes,
    pages: str = "all",
    flavor: str = "stream",
    strip_text: str = "\n",
    header_finder: bool = True,
    header_scan_rows: int = 25,
) -> pd.DataFrame:
    """
    Try camelot first; if unavailable, fall back to pdfplumber.
    If both are missing, raise a RuntimeError with install instructions.
    """

    # Try Camelot (preferred for digital table extraction)
    try:
        import camelot  # type: ignore

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name

        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor, strip_text=strip_text)
            if tables.n == 0:
                raise RuntimeError(
                    "No tables were detected by Camelot. Try switching flavor stream/lattice or adjusting pages."
                )

            cleaned = []
            for t in tables:
                df = t.df.copy()
                df.columns = [str(c) for c in df.columns]

                if header_finder:
                    hidx = detect_header_row_index(df, scan_rows=header_scan_rows)
                    if hidx is not None:
                        df = apply_header_row(df, hidx)

                cleaned.append(df)

            out = pd.concat(cleaned, ignore_index=True)
            out.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in out.columns]
            return out

        finally:
            try:
                os.remove(pdf_path)
            except Exception:
                pass

    except Exception:
        # Camelot not available or failed — try pdfplumber as a graceful fallback
        try:
            import pdfplumber  # type: ignore

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                pdf_path = tmp.name

            cleaned: List[pd.DataFrame] = []
            with pdfplumber.open(pdf_path) as doc:
                page_idxs = _parse_pages(pages, len(doc.pages))
                for pi in page_idxs:
                    if pi - 1 < 0 or pi - 1 >= len(doc.pages):
                        continue
                    page = doc.pages[pi - 1]
                    tables = page.extract_tables()
                    for t in tables:
                        try:
                            df = pd.DataFrame(t)
                        except Exception:
                            continue
                        df = df.astype(object).where(pd.notnull(df), None)
                        df.columns = [str(c) for c in df.columns]

                        if header_finder:
                            hidx = detect_header_row_index(df, scan_rows=header_scan_rows)
                            if hidx is not None:
                                df = apply_header_row(df, hidx)

                        cleaned.append(df)

            try:
                os.remove(pdf_path)
            except Exception:
                pass

            if not cleaned:
                raise RuntimeError(
                    "No tables were detected by pdfplumber. Try other pages or provide a CSV/Excel file."
                )

            out = pd.concat(cleaned, ignore_index=True)
            out.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in out.columns]
            return out

        except Exception:
            raise RuntimeError(
                "Camelot is required for PDF extraction (recommended). Alternatively install pdfplumber.\n"
                "Install with: pip install camelot-py[cv]  OR  pip install pdfplumber"
            )


# ---------------------------
# Core logic
# ---------------------------
def compute_thickness(
    df: pd.DataFrame,
    thickness_col: Optional[str],
    top_rl_col: Optional[str],
    bot_rl_col: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    note = ""
    if thickness_col:
        vals = to_numeric(df[thickness_col])
        med = float(np.nanmedian(vals.dropna())) if vals.dropna().size else np.nan
        if np.isfinite(med) and med > 0 and med < 20:
            vals_mm = vals * 1000.0
            note = f"Using provided thickness column: '{thickness_col}' (detected values ~{med:.3g} → converted m→mm)"
            df["Thickness_mm"] = vals_mm
        else:
            df["Thickness_mm"] = vals
            note = f"Using provided thickness column: '{thickness_col}'"
        return df, note

    if top_rl_col and bot_rl_col:
        top = to_numeric(df[top_rl_col])
        bot = to_numeric(df[bot_rl_col])
        th = top - bot
        med = float(np.nanmedian(th.dropna())) if th.dropna().size else np.nan
        if np.isfinite(med) and med > 0 and med < 20:
            th_mm = th * 1000.0
            note = f"Computed thickness = '{top_rl_col}' - '{bot_rl_col}' (detected units m → converted to mm)"
            df["Thickness_mm"] = th_mm
        else:
            df["Thickness_mm"] = th
            note = f"Computed thickness = '{top_rl_col}' - '{bot_rl_col}'"
        return df, note

    raise ValueError(
        "Thickness cannot be derived. Provide either a Thickness column or both Top RL and Underlying RL columns."
    )


def classify_points_updated(
    df: pd.DataFrame,
    D: float,
    A_upper: float,
    A_lower: float,
    U_upper: float,
    U_lower: float,
    level_col: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Updated classification:
      - Level nonconforming: LC > +A OR LC < -A
      - PHYSICALLY IMPOSSIBLE if:
           Thickness < (D + LC - U) [underlying cannot be higher than +U]
           OR (LC >= +A and Thickness < D) [HIGH & THIN]
      - LOW & THICK (LC <= -A and Thickness > D):
           MAY BE ACCEPTABLE – underlying low NCR (thickness made up in asphalt)
      - Otherwise:
           GENUINE NONCONFORMANCE – physically possible
      - Informational:
           Thickness > (D + LC + U) implies thicker than possible IF underlying within ±U
           (not impossibility because underlying may be low beyond -U).
    """
    df = df.copy()

    df["LevelConformance_mm"] = to_numeric(df[level_col])
    # Heuristic: if level conformance values look like metres (small magnitudes), convert to mm
    level_note = "LevelConformance assumed in mm"
    try:
        med_level = float(np.nanmedian(np.abs(df["LevelConformance_mm"].dropna()))) if df["LevelConformance_mm"].dropna().size else np.nan
        if np.isfinite(med_level) and med_level > 0 and med_level < 20:
            df["LevelConformance_mm"] = df["LevelConformance_mm"] * 1000.0
            level_note = "Detected LevelConformance in metres → converted to mm"
    except Exception:
        pass
    df["Thickness_mm"] = to_numeric(df["Thickness_mm"])

    # Filter: level nonconforming
    df["Is_Level_Nonconforming"] = (df["LevelConformance_mm"] > A_upper) | (
        df["LevelConformance_mm"] < -A_lower
    )

    # Thickness diffs vs design
    df["Thickness_Diff_vs_Design_mm"] = df["Thickness_mm"] - D

    # Pointwise feasible thickness bounds (based on underlying dev constraints)
    # Underlying_dev <= +U (cannot be higher than +U)
    Tmin_point = D + df["LevelConformance_mm"] - U_upper
    Tmax_if_underlying_within_tol = D + df["LevelConformance_mm"] + U_lower

    too_thin_impossible = df["Thickness_mm"] < Tmin_point

    high_and_thin = (df["LevelConformance_mm"] >= A_upper) & (df["Thickness_mm"] < D)
    low_and_thick = (df["LevelConformance_mm"] <= -A_lower) & (df["Thickness_mm"] > D)

    thick_gt_max_if_underlying_within_tol = df["Thickness_mm"] > Tmax_if_underlying_within_tol

    # Flags
    def _flag_row(i) -> str:
        f = []
        if pd.notna(df["Thickness_mm"].iloc[i]) and pd.notna(df["LevelConformance_mm"].iloc[i]):
            if too_thin_impossible.iloc[i]:
                f.append("Thickness<MinFeasible(D+LC-U)")
            if high_and_thin.iloc[i]:
                f.append("High&Thin")
            if low_and_thick.iloc[i]:
                f.append("Low&Thick_UnderlyingLowPossible")
            if thick_gt_max_if_underlying_within_tol.iloc[i]:
                f.append("Thick>MaxIfUnderlyingWithinTol(D+LC+U)")
        return ";".join(f)

    df["Flags"] = [_flag_row(i) for i in range(len(df))]

    # Assessment (3-way)
    df["Assessment"] = "GENUINE NONCONFORMANCE – physically possible"
    df.loc[low_and_thick, "Assessment"] = "MAY BE ACCEPTABLE – underlying low NCR (thickness made up in asphalt)"
    df.loc[too_thin_impossible | high_and_thin, "Assessment"] = "CHECK SURVEY ERROR – physically impossible"

    # Deliverable table: only level-nonconforming rows
    out = df[df["Is_Level_Nonconforming"]].copy()

    # Add helpful audit columns
    out["MinFeasibleThickness_mm (D+LC-U)"] = Tmin_point[df["Is_Level_Nonconforming"]].values
    out["MaxIfUnderlyingWithinTol_mm (D+LC+U)"] = Tmax_if_underlying_within_tol[df["Is_Level_Nonconforming"]].values
    out["DesignThickness_mm"] = D
    out["AsphaltTolUpper_mm"] = A_upper
    out["AsphaltTolLower_mm"] = A_lower
    out["UnderlyingTolUpper_mm"] = U_upper
    out["UnderlyingTolLower_mm"] = U_lower

    return out, level_note


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_csv_bytes_with_title(df: pd.DataFrame, title: Optional[str] = None) -> bytes:
    """Return CSV bytes with an optional title line at the top.

    Title will be placed as the first line, followed by a blank line, then the CSV data.
    """
    csv_body = df.to_csv(index=False)
    if title:
        title_line = f"Survey Report: {title}\n\n"
        return (title_line + csv_body).encode("utf-8-sig")
    return csv_body.encode("utf-8-sig")


# ---------------------------
# Sidebar inputs (MANDATORY FIRST STEP)
# ---------------------------
with st.sidebar:
    st.header("Inputs (must confirm)")

    D = st.number_input("Design asphalt thickness D (mm)", min_value=0.0, value=130.0, step=1.0)
    st.markdown("**Asphalt level tolerance (mm)** — specify separate upper (high) and lower (low) bounds")
    A_upper = st.number_input("Asphalt tolerance upper (+A, mm)", min_value=0.0, value=10.0, step=1.0)
    A_lower = st.number_input("Asphalt tolerance lower (-A, mm)", min_value=0.0, value=10.0, step=1.0)
    st.markdown("**Underlying layer tolerance (mm)** — specify separate upper (high) and lower (low) bounds")
    U_upper = st.number_input("Underlying tolerance upper (+U, mm)", min_value=0.0, value=10.0, step=1.0)
    U_lower = st.number_input("Underlying tolerance lower (-U, mm)", min_value=0.0, value=10.0, step=1.0)

    st.markdown("---")
    sign_ok = st.checkbox("Positive = high; Negative = low", value=True)

    st.markdown("---")
    upload = st.file_uploader("Upload CSV / Excel / PDF", type=["csv", "xlsx", "xls", "txt", "pdf"])

    st.markdown("---")
    st.subheader("PDF settings (digital tables)")
    pdf_flavor = st.selectbox("Camelot flavor", ["stream", "lattice"], index=0)
    pdf_pages = st.text_input("Pages", value="all", help='Examples: "all", "1", "1-3,5"')

    st.markdown("**Header row finder**")
    use_header_finder = st.checkbox("Auto-detect header row in each PDF table", value=True)
    header_scan_rows = st.slider("Scan first N rows for header", min_value=5, max_value=60, value=25, step=5)

if not upload:
    st.info("Upload a CSV/Excel/PDF containing Chainage, Offset, Level Conformance, and Thickness.")
    st.stop()

if not sign_ok:
    st.warning("This app assumes Positive=high and Negative=low. Tick the checkbox to proceed.")
    st.stop()

# ---------------------------
# Load data
# ---------------------------
try:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        raw = extract_tables_from_pdf(
            pdf_bytes=upload.getvalue(),
            pages=pdf_pages,
            flavor=pdf_flavor,
            header_finder=use_header_finder,
            header_scan_rows=header_scan_rows,
        )
        st.success("PDF tables extracted. Header row finder applied (if enabled).")
    else:
        raw = load_csv_excel(upload)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

raw = normalize_cols(raw)

st.subheader("Raw data preview (first 40 rows)")
st.dataframe(raw.head(40), use_container_width=True, height=360)

# ---------------------------
# Column mapping UI
# ---------------------------
st.subheader("1) Map columns")

chainage_guess = find_col(raw, ["Chainage", "Ch", "Chainage (m)", "Chainage_m"])
offset_guess = find_col(raw, ["Offset", "Off", "Offset (m)", "Offset_m"])
level_guess = find_col(raw, ["Level conformance", "LevelConformance", "Level conformance (mm)", "Conformance", "Level diff", "Delta", "Δ"])
thickness_guess = find_col(raw, ["Thickness", "Thickness (mm)", "AC Thickness", "Asphalt Thickness"])

col1, col2, col3 = st.columns(3)
with col1:
    chainage_col = st.selectbox(
        "Chainage column",
        options=["(none)"] + list(raw.columns),
        index=(1 + list(raw.columns).index(chainage_guess)) if chainage_guess in raw.columns else 0,
    )
with col2:
    offset_col = st.selectbox(
        "Offset column",
        options=["(none)"] + list(raw.columns),
        index=(1 + list(raw.columns).index(offset_guess)) if offset_guess in raw.columns else 0,
    )
with col3:
    level_col = st.selectbox(
        "Level conformance column (mm)",
        options=["(none)"] + list(raw.columns),
        index=(1 + list(raw.columns).index(level_guess)) if level_guess in raw.columns else 0,
    )

st.markdown("**Thickness column (mm)** — required: provide the survey Thickness column in millimetres.")
thickness_col = st.selectbox(
    "Thickness column (mm)",
    options=["(none)"] + list(raw.columns),
    index=(1 + list(raw.columns).index(thickness_guess)) if thickness_guess in raw.columns else 0,
)

if level_col == "(none)":
    st.error("You must map a Level conformance column (mm).")
    st.stop()

chainage_col = None if chainage_col == "(none)" else chainage_col
offset_col = None if offset_col == "(none)" else offset_col
level_col = level_col
thickness_col = None if thickness_col == "(none)" else thickness_col
top_rl_col = None
bot_rl_col = None

# ---------------------------
# Compute + classify
# ---------------------------
try:
    df = raw.copy()

    if not chainage_col:
        df["Chainage"] = np.nan
        chainage_col = "Chainage"
    if not offset_col:
        df["Offset"] = np.nan
        offset_col = "Offset"

    df["Chainage"] = to_numeric(df[chainage_col])
    df["Offset"] = to_numeric(df[offset_col])

    df, thickness_note = compute_thickness(df, thickness_col, top_rl_col, bot_rl_col)
    tagged, level_note = classify_points_updated(
        df,
        D=float(D),
        A_upper=float(A_upper),
        A_lower=float(A_lower),
        U_upper=float(U_upper),
        U_lower=float(U_lower),
        level_col=level_col,
    )

except Exception as e:
    st.error(f"Failed to process: {e}")
    st.stop()

# ---------------------------
# Outputs
# ---------------------------
st.subheader("2) Assumptions (explicit)")
st.write(
    {
        "Design thickness D (mm)": float(D),
        "Asphalt tolerance upper/lower (mm)": f"+{float(A_upper)}/-{float(A_lower)}",
        "Underlying tolerance upper/lower (mm)": f"+{float(U_upper)}/-{float(U_lower)}",
        "Sign convention": "Positive = high; Negative = low",
        "Hard minimum feasible thickness (per-point)": "Tmin_point = D + LevelConformance - U (underlying cannot be higher than +U)",
        "LOW & THICK handling": "Flagged as MAY BE ACCEPTABLE (subject to underlying low NCR)",
        "Thickness source": thickness_note,
        "PDF mode": f"Digital tables via Camelot ({pdf_flavor})",
        "LevelConformance source": level_note,
        "Header row finder": "Enabled" if use_header_finder else "Disabled",
    }
)

total_nonconf = len(tagged)
n_check = int((tagged["Assessment"] == "CHECK SURVEY ERROR – physically impossible").sum())
n_genuine = int((tagged["Assessment"] == "GENUINE NONCONFORMANCE – physically possible").sum())
n_maybe = int((tagged["Assessment"] == "MAY BE ACCEPTABLE – underlying low NCR (thickness made up in asphalt)").sum())

st.subheader("3) Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total level-nonconforming points", total_nonconf)
c2.metric("CHECK SURVEY ERROR", n_check)
c3.metric("GENUINE NONCONFORMANCE", n_genuine)
c4.metric("MAY BE ACCEPTABLE", n_maybe)

st.subheader("4) Tagged table (level-nonconforming points only)")
mandatory_cols = [
    "Chainage",
    "Offset",
    "LevelConformance_mm",
    "Thickness_mm",
    "Thickness_Diff_vs_Design_mm",
    "MinFeasibleThickness_mm (D+LC-U)",
    "MaxIfUnderlyingWithinTol_mm (D+LC+U)",
    "Assessment",
    "Flags",
]
for col in mandatory_cols:
    if col not in tagged.columns:
        tagged[col] = np.nan

st.dataframe(tagged[mandatory_cols], use_container_width=True, height=560)

st.subheader("5) Download")
st.download_button(
    label="Download tagged CSV",
    data=df_to_csv_bytes_with_title(
        tagged[mandatory_cols], title=os.path.splitext(upload.name)[0]
    ),
    file_name=f"{re.sub(r'[^A-Za-z0-9_-]', '_', os.path.splitext(upload.name)[0])}_Asphalt_NCR_Checks_Tagged_Level_Nonconforming_Points.csv",
    mime="text/csv",
)

st.caption(
    "Interpretation: "
    "CHECK SURVEY ERROR = physically impossible given underlying cannot be above +U. "
    "MAY BE ACCEPTABLE = low & thick, consistent with underlying low (potentially accepted NCR)."
)
