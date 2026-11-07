"""Environment data: game schedule/results (nflverse) and NFL.com weekly power rankings."""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "nflcom"
GAMES_CSV = DATA / "games.csv"
RANKINGS_CSV = DATA / "power_rankings_nflcom.csv"
NW_TIERS_CSV = DATA / "nick_wright_tiers.csv"

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFLCOM_URL = "https://www.nfl.com/news/nfl-power-rankings-week-{week}-{season}-nfl-season"

# NFL.com full team names -> nflverse abbreviations.
TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
# NFL.com logo codes that differ from nflverse.
LOGO_FIX = {"AZ": "ARI"}


def load_games(refresh: bool = False) -> pd.DataFrame:
    """Regular-season games. Uses the committed snapshot unless refresh=True."""
    if refresh or not GAMES_CSV.exists():
        GAMES_CSV.write_bytes(requests.get(GAMES_URL, timeout=60).content)
    g = pd.read_csv(GAMES_CSV)
    g = g[g.game_type == "REG"].copy()
    return g[["game_id", "season", "week", "gameday", "gametime", "location", "away_team", "home_team",
              "away_score", "home_score", "result", "spread_line"]]


def _fetch_page(season: int, week: int, refresh: bool = False) -> str | None:
    path = RAW / f"{season}_w{week:02d}.html"
    if path.exists() and not refresh:
        return path.read_text()
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(NFLCOM_URL.format(season=season, week=week),
                     headers={"User-Agent": "Mozilla/5.0 (CS4300 course project)"}, timeout=60)
    time.sleep(1.0)  # be polite to nfl.com
    if r.status_code != 200:
        return None
    path.write_text(r.text)
    return r.text


def parse_rankings(page: str, season: int, week: int) -> pd.DataFrame:
    """Extract (rank, team, record) from an NFL.com Power Rankings article.

    Each entry in the article body is rendered as: "Rank" / N / ... / <Full Team Name> / <W-L>.
    The result is validated (32 unique teams, ranks 1..32) and cross-checked against
    the team logos the page preloads, which must appear in the same relative order
    (the page sometimes omits the top few logos, so this is an ordering check).
    """
    canon = re.search(r'rel="canonical" href="([^"]*)"', page)
    if not canon or f"week-{week}-{season}-nfl-season" not in canon.group(1):
        raise ValueError(f"{season} w{week}: page is not the expected article")

    body = page[page.find("<body"):]
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", body, flags=re.S)
    lines = [l.strip() for l in html.unescape(re.sub(r"<[^>]+>", "\n", body)).split("\n")]
    lines = [l for l in lines if l]

    rows = []
    for i, line in enumerate(lines[:-1]):
        if line == "Rank" and lines[i + 1].isdigit():
            rank = int(lines[i + 1])
            for j in range(i + 2, min(i + 12, len(lines))):
                if lines[j] in TEAMS:
                    rec = lines[j + 1] if re.fullmatch(r"\d+-\d+(-\d+)?", lines[j + 1]) else ""
                    rows.append({"season": season, "week": week, "rank": rank,
                                 "team": TEAMS[lines[j]], "record": rec})
                    break
    df = pd.DataFrame(rows).drop_duplicates(["rank"])
    if len(df) != 32 or set(df["rank"]) != set(range(1, 33)) or df.team.nunique() != 32:
        raise ValueError(f"{season} w{week}: parsed {len(df)} ranked teams, expected 32")

    logos = []
    for t in re.findall(r"clubs/logos/([A-Z]+)\"", page):
        t = LOGO_FIX.get(t, t)
        if t in logos:
            break
        logos.append(t)
    order = df.sort_values("rank").team.tolist()
    if len(logos) < 20 or [t for t in order if t in logos] != logos:
        raise ValueError(f"{season} w{week}: body ranking disagrees with logo order")
    return df.sort_values("rank").reset_index(drop=True)


def build_rankings(seasons: list[int], weeks=range(1, 19), refresh: bool = False) -> pd.DataFrame:
    """Scrape + parse every available week and write data/power_rankings_nflcom.csv."""
    frames, log = [], []
    for s in seasons:
        for w in weeks:
            page = _fetch_page(s, w, refresh)
            if page is None:
                log.append(f"{s} w{w}: not published")
                continue
            try:
                frames.append(parse_rankings(page, s, w))
                log.append(f"{s} w{w}: ok")
            except ValueError as e:
                log.append(f"skip {e}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RANKINGS_CSV, index=False)
    return df, log


def load_rankings() -> pd.DataFrame:
    return pd.read_csv(RANKINGS_CSV)


def tier_ranks(tiers: pd.DataFrame) -> pd.DataFrame:
    """Turn Nick Wright's tiers (1 = top tier) into ranks, ties sharing the tier's best rank.

    A team's rank is 1 + the number of teams in higher tiers, so tier sizes 1, 2, 3 give
    ranks 1, 2, 2, 4, 4, 4. Teams in the tier named "Eliminated" share the average of the
    ranks they span instead (two teams at 31-32 both get 31.5).
    Each week must list all 32 teams exactly once.
    """
    frames = []
    for (season, week), wk in tiers.groupby(["season", "week"], sort=True):
        where = f"{NW_TIERS_CSV.name} {season} w{week}"
        if wk.tier.isna().any():
            raise ValueError(f"{where}: {int(wk.tier.isna().sum())} teams have no tier")
        if sorted(wk.team) != sorted(TEAMS.values()):
            missing = sorted(set(TEAMS.values()) - set(wk.team))
            extra = sorted(set(wk.team) - set(TEAMS.values()))
            dupes = sorted(wk.team[wk.team.duplicated()])
            raise ValueError(f"{where}: needs all 32 teams once "
                             f"(missing {missing}, unknown {extra}, duplicated {dupes})")
        if (wk.tier % 1 != 0).any() or (wk.tier < 1).any():
            raise ValueError(f"{where}: tiers must be whole numbers starting at 1")
        wk = wk.astype({"tier": int})
        best = wk.tier.map(lambda t: 1.0 + (wk.tier < t).sum())
        out = wk.tier_name.fillna("").str.strip().str.lower() == "eliminated"
        wk["rank"] = best.where(~out, best + (out.sum() - 1) / 2)
        frames.append(wk)
    cols = ["season", "week", "tier", "tier_name", "team", "rank"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols].sort_values(["season", "week", "rank"])


def load_nick_tiers() -> pd.DataFrame:
    """Hand-entered Nick Wright tiers as ranks. Weeks with no tiers filled in are skipped."""
    if not NW_TIERS_CSV.exists():
        return tier_ranks(pd.DataFrame(columns=["season", "week", "tier", "tier_name", "team"]))
    t = pd.read_csv(NW_TIERS_CSV, skipinitialspace=True)
    t["team"] = t.team.str.strip()
    entered = t.groupby(["season", "week"]).tier.transform(lambda s: s.notna().any())
    return tier_ranks(t[entered])
