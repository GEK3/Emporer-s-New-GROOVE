import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

st.set_page_config(page_title="Hitting Profiles", page_icon="⚾", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 1.3rem !important; font-weight: 700 !important; margin-bottom: 0 !important; }
    .subtitle { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
    label { font-size: 0.7rem !important; color: #aaa !important;
            text-transform: uppercase; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)

METRIC_COLS   = ["GROOVE","Damage/BBE","Selectivity (%)","Hittable Pitch Take (%)",
                 "Chase (%)","Z-Contact (%)","Whiff vs. Secondaries (%)","Z-Swing (%)","Zone (%)"]
LOWER_BETTER  = {"Hittable Pitch Take (%)","Chase (%)","Whiff vs. Secondaries (%)"}
COL_LABELS    = {
    "GROOVE":"GROOVE","Damage/BBE":"Damage/BBE","Selectivity (%)":"Selectivity",
    "Hittable Pitch Take (%)":"Hittable Take","Chase (%)":"Chase",
    "Z-Contact (%)":"Z-Contact","Whiff vs. Secondaries (%)":"Whiff vs Sec",
    "Z-Swing (%)":"Z-Swing","Zone (%)":"Zone",
}
COL_HELP = {
    "GROOVE":"Pitch-level run value metric grading offensive execution. Decision quality × execution outcome. Contact uses xwOBA (outcome-independent). Heart-zone whiffs penalized most. Derived from 2024-2025 Statcast backtests.",
    "Damage/BBE":"% of contact events with xwOBA ≥ 0.350 (above-average quality contact).",
    "Selectivity (%)":"Good Takes / Good Decisions. Higher = more selective approach.",
    "Hittable Pitch Take (%)":"Hittable pitches taken / all takes. Lower = fewer free swings left on table.",
    "Chase (%)":"O-Swing%. Lower = better.",
    "Z-Contact (%)":"Contact rate on zone pitches.",
    "Whiff vs. Secondaries (%)":"Whiff/swing vs breaking balls and offspeed. Lower = harder to put away.",
    "Z-Swing (%)":"Swing rate on zone pitches.",
    "Zone (%)":"% of pitches in the zone to this hitter.",
}
DAMAGE_XWOBA = 0.350
OF_POSITIONS = {"OF","LF","CF","RF"}
POS_ORDER    = ["All","C","1B","2B","3B","SS","OF","DH"]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_pitch_scores():
    p = Path(__file__).parent / "data" / "pitch_scores.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=3600)
def load_season_leaderboard():
    for name in ["leaderboard_Season.csv", "leaderboard.csv"]:
        p = Path(__file__).parent / "data" / name
        if p.exists():
            df = pd.read_csv(p)
            for col in METRIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "PA" in df.columns:
                df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0).astype(int)
            return df
    return None


def aggregate_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized aggregation from pitch_scores."""
    g = scores.groupby(["batter","Name","Team","Pos"])
    agg = g.agg(
        PA             =("is_pa_end",      "sum"),
        pitch_rv_sum   =("pitch_rv",       "sum"),
        pitch_rv_count =("pitch_rv",       "count"),
        good_takes     =("is_good_take",   "sum"),
        good_swings    =("is_good_swing",  "sum"),
        hittable_takes =("is_hittable_take","sum"),
        total_takes    =("is_take",        "sum"),
        chase_swings   =("is_chase_swing", "sum"),
        ooz_pitches    =("is_ooz_pitch",   "sum"),
        zone_swings    =("is_zone_swing",  "sum"),
        zone_pitches   =("is_zone_pitch",  "sum"),
        zone_contacts  =("is_zone_contact","sum"),
        sec_swings     =("is_sec_swing",   "sum"),
        sec_whiffs     =("is_sec_whiff",   "sum"),
        damage_count   =("xwoba",          lambda x: (x >= DAMAGE_XWOBA).sum()),
        air_balls      =("launch_angle",   lambda x: (x > 0).sum()),
    ).reset_index()

    agg["GROOVE"]                    = (agg["pitch_rv_sum"] / agg["pitch_rv_count"].replace(0,np.nan) * 100).round(2)
    gd = agg["good_swings"] + agg["good_takes"]
    agg["Selectivity (%)"]           = (agg["good_takes"]     / gd.replace(0,np.nan) * 100).round(1)
    agg["Hittable Pitch Take (%)"]   = (agg["hittable_takes"] / agg["total_takes"].replace(0,np.nan) * 100).round(1)
    agg["Chase (%)"]                 = (agg["chase_swings"]   / agg["ooz_pitches"].replace(0,np.nan) * 100).round(1)
    agg["Z-Contact (%)"]             = (agg["zone_contacts"]  / agg["zone_swings"].replace(0,np.nan) * 100).round(1)
    agg["Z-Swing (%)"]               = (agg["zone_swings"]    / agg["zone_pitches"].replace(0,np.nan) * 100).round(1)
    agg["Zone (%)"]                  = (agg["zone_pitches"]   / agg["pitch_rv_count"].replace(0,np.nan) * 100).round(1)
    agg["Whiff vs. Secondaries (%)"] = (agg["sec_whiffs"]     / agg["sec_swings"].replace(0,np.nan) * 100).round(1)
    agg["Damage/BBE"]                = (agg["damage_count"]   / agg["air_balls"].replace(0,np.nan) * 100).round(1)
    agg["PA"]                        = agg["PA"].astype(int)

    keep = ["Name","Team","Pos","PA","GROOVE","Damage/BBE",
            "Selectivity (%)","Hittable Pitch Take (%)","Chase (%)",
            "Z-Contact (%)","Whiff vs. Secondaries (%)","Z-Swing (%)","Zone (%)"]
    return agg[[c for c in keep if c in agg.columns]].sort_values("GROOVE", ascending=False).reset_index(drop=True)


def compute_percentiles(df):
    pct = df[["Name","Team"]].copy()
    for col in ["Pos","PA"]:
        if col in df.columns: pct[col] = df[col]
    for col in METRIC_COLS:
        if col not in df.columns: continue
        series = df[col].dropna()
        lower  = col in LOWER_BETTER
        pct[col] = df[col].apply(
            lambda v: np.nan if pd.isna(v) else
            float(np.mean(series < v)*100) if not lower else float(np.mean(series > v)*100)
        )
    return pct


def safe_range(df, col, dec=1):
    if col not in df.columns or df[col].dropna().empty: return 0.0, 1.0
    lo, hi = float(round(df[col].min(), dec)), float(round(df[col].max(), dec))
    return (lo, hi) if lo != hi else (lo, lo+1.0)


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    c1, c2 = st.columns([3,1])
    with c1:
        st.title("⚾ Hitting Profiles")
        st.markdown('<p class="subtitle">Plate discipline & damage metrics — GROOVE framework</p>',
                    unsafe_allow_html=True)
    with c2:
        for name in ["leaderboard_Season.csv","leaderboard.csv"]:
            p = Path(__file__).parent / "data" / name
            if p.exists():
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                st.caption(f"Updated: {mtime.strftime('%b %d, %Y')}")
                break

    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    pitch_scores = load_pitch_scores()
    has_parquet  = pitch_scores is not None

    if has_parquet:
        min_date = pitch_scores["game_date"].min().date()
        max_date = pitch_scores["game_date"].max().date()
    else:
        df_season = load_season_leaderboard()
        min_date  = datetime.date(datetime.date.today().year, 3, 20)
        max_date  = datetime.date.today()

    # ══════════════════════════════════════════════════════════════
    # FILTER PANEL — Row 1: text/dropdowns
    # ══════════════════════════════════════════════════════════════
    f1, f2, f3, f4, f5 = st.columns([2.2, 1.2, 1.2, 1.5, 0.8])

    with f1:
        search = st.text_input("Batter Search", placeholder="e.g. Judge, NYY")

    # Load season leaderboard for dropdown options
    df_season = load_season_leaderboard()
    teams_list = ["All"] + (sorted(df_season["Team"].dropna().unique().tolist()) if df_season is not None else [])
    avail_pos  = [p for p in POS_ORDER if p == "All" or
                  (df_season is not None and
                   (p in df_season.get("Pos", pd.Series()).values or
                    (p == "OF" and any(x in df_season.get("Pos", pd.Series()).values for x in OF_POSITIONS))))]

    with f2:
        team_filter = st.selectbox("Batter Team", teams_list)

    with f3:
        pos_filter = st.selectbox("Position", avail_pos)

    with f4:
        mode     = st.radio("View", ["Raw values","Percentiles"], horizontal=True)
        mode_key = "raw" if mode == "Raw values" else "pct"

    with f5:
        pa_min = st.number_input("Min PA", min_value=0, max_value=700, value=50, step=25)

    # ── Row 2: date slider + metric sliders ───────────────────────────────────
    d1, d2 = st.columns([2, 3])

    with d1:
        date_range = st.slider(
            "Game Date",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MM/DD/YY",
        )

    # Compute leaderboard for selected date range
    if has_parquet:
        filtered_scores = pitch_scores[
            (pitch_scores["game_date"].dt.date >= date_range[0]) &
            (pitch_scores["game_date"].dt.date <= date_range[1])
        ]
        with st.spinner("Computing…"):
            df_raw = aggregate_from_scores(filtered_scores)
    else:
        df_raw = df_season.copy() if df_season is not None else pd.DataFrame()

    with d2:
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            lo, hi = safe_range(df_raw, "GROOVE", 2)
            groove_range = st.slider("GROOVE", lo, hi, (lo,hi), step=0.05, format="%.2f")
        with s2:
            lo, hi = safe_range(df_raw, "Damage/BBE", 1)
            dmg_range = st.slider("Damage/BBE", lo, hi, (lo,hi), step=1.0, format="%.1f")
        with s3:
            lo, hi = safe_range(df_raw, "Chase (%)", 1)
            chase_range = st.slider("Chase %", lo, hi, (lo,hi), step=0.5, format="%.1f")
        with s4:
            lo, hi = safe_range(df_raw, "Z-Contact (%)", 1)
            zcon_range = st.slider("Z-Contact %", lo, hi, (lo,hi), step=0.5, format="%.1f")
        with s5:
            lo, hi = safe_range(df_raw, "Whiff vs. Secondaries (%)", 1)
            whiff_range = st.slider("Whiff vs Sec %", lo, hi, (lo,hi), step=0.5, format="%.1f")

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # APPLY FILTERS
    # ══════════════════════════════════════════════════════════════
    df = df_raw.copy()

    if search:
        mask = (df["Name"].str.lower().str.contains(search.lower(), na=False) |
                df["Team"].str.lower().str.contains(search.lower(), na=False))
        df = df[mask]

    if team_filter != "All" and "Team" in df.columns:
        df = df[df["Team"] == team_filter]

    if pos_filter != "All" and "Pos" in df.columns:
        if pos_filter == "OF":
            df = df[df["Pos"].isin(OF_POSITIONS)]
        else:
            df = df[df["Pos"] == pos_filter]

    if "PA" in df.columns:
        df = df[df["PA"] >= pa_min]

    for col, rng in [("GROOVE", groove_range), ("Damage/BBE", dmg_range),
                     ("Chase (%)", chase_range), ("Z-Contact (%)", zcon_range),
                     ("Whiff vs. Secondaries (%)", whiff_range)]:
        if col in df.columns:
            df = df[df[col].between(*rng)]

    df = df.sort_values("GROOVE", ascending=False).reset_index(drop=True)
    st.caption(f"Showing {len(df)} players  ·  {date_range[0].strftime('%b %d')} – {date_range[1].strftime('%b %d, %Y')}")

    # ══════════════════════════════════════════════════════════════
    # TABLE
    # ══════════════════════════════════════════════════════════════
    display_cols = ["Name","Team"]
    for c in ["Pos","PA"]:
        if c in df.columns: display_cols.append(c)
    display_cols += [c for c in METRIC_COLS if c in df.columns]

    if mode_key == "pct":
        pct_df = compute_percentiles(df_raw)
        pct_df = pct_df[pct_df["Name"].isin(df["Name"])].sort_values("GROOVE", ascending=False).reset_index(drop=True)
        display_df = pct_df[[c for c in display_cols if c in pct_df.columns]]
    else:
        display_df = df[[c for c in display_cols if c in df.columns]]

    pct_style = compute_percentiles(df_raw)
    pct_style = pct_style[pct_style["Name"].isin(display_df["Name"])].sort_values("GROOVE", ascending=False).reset_index(drop=True)

    fmt_df = display_df.copy()
    for col in METRIC_COLS:
        if col not in fmt_df.columns: continue
        if mode_key == "raw":
            fmt_df[col] = fmt_df[col].apply(
                lambda v: (f"{v:.2f}" if col == "GROOVE" else f"{v:.1f}") if pd.notna(v) else "—")
        else:
            fmt_df[col] = fmt_df[col].apply(lambda v: f"{round(v)}" if pd.notna(v) else "—")

    fmt_df = fmt_df.rename(columns=COL_LABELS)

    styler = fmt_df.style
    for col in METRIC_COLS:
        if col not in pct_style.columns: continue
        dcol = COL_LABELS.get(col, col)
        if dcol not in fmt_df.columns: continue
        gmap = pct_style[col].values
        if len(gmap) == len(fmt_df):
            styler = styler.background_gradient(cmap="RdYlGn", subset=[dcol], vmin=0, vmax=100, gmap=gmap)

    styler = styler.set_properties(**{"text-align":"right","font-size":"12px"})
    styler = styler.set_properties(subset=["Name"], **{"text-align":"left","font-weight":"600"})
    if "Team" in fmt_df.columns:
        styler = styler.set_properties(subset=["Team"], **{"text-align":"left","color":"#888"})

    st.dataframe(styler, use_container_width=True, height=600)

    with st.expander("Metric definitions"):
        for col, desc in COL_HELP.items():
            st.markdown(f"**{col}** — {desc}")

    st.divider()
    with st.expander("Data controls"):
        st.markdown("Leaderboard refreshes daily via GitHub Actions.")
        if st.button("Clear cached data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
