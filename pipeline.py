"""
pipeline.py — Hitting Profiles data pipeline

Saves two files:
  data/pitch_scores.parquet  — pitch-level indicators for every hitter PA
  data/player_info.csv       — name, team, position per batter ID

The app loads these and recomputes metrics for any date range instantly.

Usage:
    python pipeline.py                         # full current season
    python pipeline.py --start 2026-03-20 --end 2026-04-25
"""

import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
from pybaseball import statcast

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

DATA_DIR = Path(__file__).parent / "data"

# ── GROOVE weights (empirically derived from 2024-2025 Statcast backtests) ───
GROOVE_WEIGHTS = {
    "A+":        0.3666,
    "D":         0.0442,
    "B0":       -0.0138,
    "A~":       -0.0379,
    "B~":       -0.0399,
    "C":        -0.0559,
    "B-_inzone":-0.0971,
    "A-_heart": -0.1001,
    "A-_inzone":-0.1086,
    "B-_shadow":-0.1179,
    "A0":       -0.1617,
    "A-_shadow":-0.1719,
    "B-_out":   -0.1179,
    "A-_unknown":-0.1086,
    "B-_unknown":-0.1086,
}

LEAGUE_XWOBA  = 0.320
WOBA_SCALE    = 1.157
DAMAGE_XWOBA  = 0.350

ZONE_HEART    = {5}
ZONE_INZONE   = {1, 2, 3, 4, 6, 7, 8, 9}
ZONE_SHADOW   = {11, 12, 13, 14}
# TWP (two-way player) intentionally NOT in PITCHER_POS — Ohtani bats
PITCHER_POS   = {"P", "SP", "RP"}

SWING_DESC    = {"hit_into_play","swinging_strike","swinging_strike_blocked",
                 "foul","foul_tip","foul_bunt","missed_bunt"}
TAKE_DESC     = {"called_strike","ball","blocked_ball","pitchout","hit_by_pitch"}
WHIFF_DESC    = {"swinging_strike","swinging_strike_blocked","missed_bunt"}
FOUL_DESC     = {"foul","foul_tip","foul_bunt"}
CONTACT_DESC  = {"hit_into_play"}
SECONDARY     = {"SL","CU","KC","SV","ST","CUO","SLO","CH","FS","FO","SC","CS"}
PA_EVENTS     = {"strikeout","single","double","triple","home_run","walk","hit_by_pitch",
                 "field_out","force_out","grounded_into_double_play","sac_fly","sac_bunt",
                 "fielders_choice","fielders_choice_out","double_play",
                 "strikeout_double_play","intent_walk","field_error","other_out","catcher_interf"}


def season_dates():
    year = datetime.date.today().year
    return f"{year}-03-20", datetime.date.today().strftime("%Y-%m-%d")


def compute_spray_angle(df):
    hc_x = df["hc_x"].fillna(125.42) - 125.42
    hc_y = 198.27 - df["hc_y"].fillna(198.27)
    angle = np.degrees(np.arctan2(hc_x, hc_y))
    lhh = df["stand"] == "L"
    angle[lhh] = -angle[lhh]
    return angle


def get_player_info(mlbam_ids):
    import urllib.request, json
    results = {}
    for i in range(0, len(mlbam_ids), 100):
        batch   = mlbam_ids[i:i+100]
        ids_str = ",".join(str(x) for x in batch)
        url = (f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
               f"&fields=people,id,fullName,primaryPosition,abbreviation")
        try:
            with urllib.request.urlopen(url) as r:
                data = json.loads(r.read())
            for p in data["people"]:
                results[p["id"]] = {
                    "name": p["fullName"],
                    "pos":  p.get("primaryPosition", {}).get("abbreviation", "?")
                }
        except Exception as e:
            print(f"  API error: {e}")
    return results


def get_hitter_ids(raw):
    batter_ids = raw["batter"].dropna().astype(int).unique().tolist()
    print(f"  Looking up {len(batter_ids)} batter IDs via MLB API...")
    info = get_player_info(batter_ids)
    pos_counts = {}
    for v in info.values():
        pos_counts[v["pos"]] = pos_counts.get(v["pos"], 0) + 1
    print(f"  Positions: {dict(sorted(pos_counts.items(), key=lambda x: -x[1])[:10])}")
    hitters  = {k for k, v in info.items() if v["pos"] not in PITCHER_POS}
    pitchers = {k for k, v in info.items() if v["pos"] in PITCHER_POS}
    print(f"  Excluded {len(pitchers)} pitchers, {len(hitters)} hitters remaining")
    return hitters, info


def compute_pitch_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-pitch indicators for all metrics.
    Returns one row per pitch with game_date, batter, and all indicator columns.
    """
    p = raw.dropna(subset=["plate_x","plate_z","balls","strikes","delta_run_exp"]).copy()
    p["is_swing"] = p["description"].isin(SWING_DESC)
    p["is_take"]  = p["description"].isin(TAKE_DESC)
    p = p[p["is_swing"] | p["is_take"]].copy()

    # ── Advantage (swing_rv - take_rv) ────────────────────────────────────────
    p["px_bin"] = pd.cut(p["plate_x"], bins=np.linspace(-2,2,13), labels=False)
    p["pz_bin"] = pd.cut(p["plate_z"], bins=np.linspace(0,5,13),  labels=False)
    key = ["balls","strikes","px_bin","pz_bin"]

    swing_rv = (p[p["is_swing"]].groupby(key)["delta_run_exp"]
                .mean().reset_index().rename(columns={"delta_run_exp":"swing_rv"}))
    take_rv  = (p[p["is_take"]].groupby(key)["delta_run_exp"]
                .mean().reset_index().rename(columns={"delta_run_exp":"take_rv"}))

    p = p.merge(swing_rv, on=key, how="left")
    p = p.merge(take_rv,  on=key, how="left")
    p["swing_rv"]  = p["swing_rv"].fillna(0)
    p["take_rv"]   = p["take_rv"].fillna(0)
    p["advantage"] = p["swing_rv"] - p["take_rv"]

    # ── Zone classification ───────────────────────────────────────────────────
    def zlbl(z):
        if z in ZONE_HEART:   return "heart"
        if z in ZONE_INZONE:  return "inzone"
        if z in ZONE_SHADOW:  return "shadow"
        return "out"

    p["zone_lbl"] = p["zone"].apply(lambda z: zlbl(z) if pd.notna(z) else "unknown")
    p["in_zone"]  = p["zone"].isin(ZONE_HEART | ZONE_INZONE)

    # ── Tier classification ───────────────────────────────────────────────────
    def tier(row):
        adv, sw, desc, zl = row["advantage"], row["is_swing"], row["description"], row["zone_lbl"]
        if sw:
            correct = adv > 0
            if desc in CONTACT_DESC:  return "A_contact" if correct else "B0"
            elif desc in FOUL_DESC:   return "A~"  if correct else "B~"
            elif desc in WHIFF_DESC:  return f"A-_{zl}" if correct else f"B-_{zl}"
            else:                     return "A~"  if correct else "B~"
        else:
            return "D" if adv <= 0 else "C"

    p["tier"] = p.apply(tier, axis=1)

    # ── GROOVE run value per pitch ────────────────────────────────────────────
    contact_mask = p["description"].isin(CONTACT_DESC) & p["estimated_woba_using_speedangle"].notna()
    dmg_mask     = p["estimated_woba_using_speedangle"] >= DAMAGE_XWOBA

    p.loc[contact_mask &  dmg_mask, "tier"] = "A+"
    p.loc[contact_mask & ~dmg_mask, "tier"] = "A0"

    p["pitch_rv"] = np.where(
        contact_mask,
        (p["estimated_woba_using_speedangle"] - LEAGUE_XWOBA) / WOBA_SCALE,
        p["tier"].map(GROOVE_WEIGHTS)
    )

    # ── Decision indicators (for Selectivity / Hittable Pitch Take) ──────────
    p["is_good_take"]     = (~p["is_swing"]) & (p["advantage"] <= 0)   # D
    p["is_good_swing"]    = p["is_swing"]    & (p["advantage"] > 0)    # A
    p["is_hittable_take"] = (~p["is_swing"]) & (p["advantage"] > 0)    # C

    # ── Zone / chase indicators ───────────────────────────────────────────────
    p["is_chase_swing"]   = p["is_swing"]  & ~p["in_zone"]
    p["is_ooz_pitch"]     = ~p["in_zone"]
    p["is_zone_swing"]    = p["is_swing"]  &  p["in_zone"]
    p["is_zone_pitch"]    = p["in_zone"]
    p["is_zone_contact"]  = p["is_zone_swing"] & p["description"].isin(CONTACT_DESC | {"foul","foul_tip"})

    # ── Secondary whiff indicators ────────────────────────────────────────────
    p["is_sec_pitch"]     = p["pitch_type"].isin(SECONDARY)
    p["is_sec_swing"]     = p["is_sec_pitch"] & p["is_swing"]
    p["is_sec_whiff"]     = p["is_sec_pitch"] & p["description"].isin(WHIFF_DESC)

    # ── PA end indicator ──────────────────────────────────────────────────────
    p["is_pa_end"] = p["events"].isin(PA_EVENTS)

    # ── Batted ball indicators (for Damage/BBE) ───────────────────────────────
    # We'll compute damage from the raw batted ball data separately
    # Store xwOBA for contact events so we can compute damage in app
    p["xwoba"]       = p["estimated_woba_using_speedangle"]
    p["is_contact"]  = p["description"].isin(CONTACT_DESC)
    p["launch_speed"] = p["launch_speed"] if "launch_speed" in p.columns else np.nan
    p["launch_angle"] = p["launch_angle"] if "launch_angle" in p.columns else np.nan

    keep = [
        "game_date","batter",
        "pitch_rv",
        "is_good_take","is_good_swing","is_hittable_take","is_swing","is_take",
        "is_chase_swing","is_ooz_pitch",
        "is_zone_swing","is_zone_pitch","is_zone_contact",
        "is_sec_swing","is_sec_whiff",
        "is_pa_end","is_contact",
        "xwoba","launch_speed","launch_angle","in_zone",
    ]
    keep = [c for c in keep if c in p.columns]
    return p[keep].copy()


def run(start: str, end: str) -> None:
    print(f"Pulling Statcast data {start} → {end}…")
    raw = statcast(start, end)
    print(f"  {len(raw):,} pitches")
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    # Team assignment
    raw["batter_team"] = np.where(
        raw["inning_topbot"] == "Top", raw["away_team"], raw["home_team"]
    )

    print("Filtering to hitters...")
    hitter_ids, player_info = get_hitter_ids(raw)
    raw = raw[raw["batter"].isin(hitter_ids)].copy()

    # ── Save player info (name, team, position) ───────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    player_df = pd.DataFrame([
        {"batter": k, "Name": v["name"], "Pos": v["pos"]}
        for k, v in player_info.items() if v["pos"] not in PITCHER_POS
    ])
    teams = (raw.sort_values("game_date").groupby("batter")["batter_team"].last()
             .reset_index().rename(columns={"batter_team": "Team"}))
    player_df = player_df.merge(teams, on="batter", how="left")
    player_df.to_csv(DATA_DIR / "player_info.csv", index=False)
    print(f"  Saved player_info.csv ({len(player_df)} players)")

    # ── Compute and save pitch-level scores ───────────────────────────────────
    print("Computing pitch-level scores…")
    scores = compute_pitch_scores(raw)
    scores = scores.merge(player_df[["batter","Name","Team","Pos"]], on="batter", how="left")
    scores["game_date"] = pd.to_datetime(scores["game_date"])

    out = DATA_DIR / "pitch_scores.parquet"
    scores.to_parquet(out, index=False)
    print(f"  Saved pitch_scores.parquet ({len(scores):,} rows)")

    # ── Also save a full-season leaderboard CSV for backwards compat ──────────
    print("Computing season leaderboard…")
    lb = aggregate_leaderboard(scores)
    lb.to_csv(DATA_DIR / "leaderboard_Season.csv", index=False)
    print(f"  Saved leaderboard_Season.csv ({len(lb)} rows)")


def aggregate_leaderboard(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch_scores to per-player leaderboard."""
    g = scores.groupby(["batter","Name","Team","Pos"])

    agg = g.agg(
        PA              =("is_pa_end",     "sum"),
        pitch_rv_sum    =("pitch_rv",      "sum"),
        pitch_rv_count  =("pitch_rv",      "count"),
        good_takes      =("is_good_take",  "sum"),
        good_swings     =("is_good_swing", "sum"),
        hittable_takes  =("is_hittable_take","sum"),
        total_takes     =("is_take",       "sum"),
        chase_swings    =("is_chase_swing","sum"),
        ooz_pitches     =("is_ooz_pitch",  "sum"),
        zone_swings     =("is_zone_swing", "sum"),
        zone_pitches    =("is_zone_pitch", "sum"),
        zone_contacts   =("is_zone_contact","sum"),
        sec_swings      =("is_sec_swing",  "sum"),
        sec_whiffs      =("is_sec_whiff",  "sum"),
        damage_count    =("xwoba",         lambda x: (x >= DAMAGE_XWOBA).sum()),
        air_balls       =("launch_angle",  lambda x: (x > 0).sum()),
    ).reset_index()

    agg["GROOVE"]                  = (agg["pitch_rv_sum"] / agg["pitch_rv_count"] * 100).round(2)
    good_dec = agg["good_swings"] + agg["good_takes"]
    agg["Selectivity (%)"]         = (agg["good_takes"] / good_dec.replace(0, np.nan) * 100).round(1)
    agg["Hittable Pitch Take (%)"] = (agg["hittable_takes"] / agg["total_takes"].replace(0, np.nan) * 100).round(1)
    agg["Chase (%)"]               = (agg["chase_swings"] / agg["ooz_pitches"].replace(0, np.nan) * 100).round(1)
    agg["Z-Contact (%)"]           = (agg["zone_contacts"] / agg["zone_swings"].replace(0, np.nan) * 100).round(1)
    agg["Z-Swing (%)"]             = (agg["zone_swings"] / agg["zone_pitches"].replace(0, np.nan) * 100).round(1)
    agg["Zone (%)"]                = (agg["zone_pitches"] / agg["pitch_rv_count"].replace(0, np.nan) * 100).round(1)
    agg["Whiff vs. Secondaries (%)"] = (agg["sec_whiffs"] / agg["sec_swings"].replace(0, np.nan) * 100).round(1)
    agg["Damage/BBE"]              = (agg["damage_count"] / agg["air_balls"].replace(0, np.nan) * 100).round(1)

    keep = ["Name","Team","Pos","PA","GROOVE","Damage/BBE",
            "Selectivity (%)","Hittable Pitch Take (%)","Chase (%)",
            "Z-Contact (%)","Whiff vs. Secondaries (%)","Z-Swing (%)","Zone (%)"]
    return agg[keep].sort_values("GROOVE", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()
    start, end = args.start, args.end
    if not start or not end:
        start, end = season_dates()
    run(start, end)
