import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path('/mnt/data')
PBP_DIR = BASE_DIR / 'tp' / 'tennis_pointbypoint-master'
ATP_DIR = BASE_DIR / 'ta' / 'tennis_atp-master'
OUT_DIR = BASE_DIR / 'dsa210_outputs'
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = [3, 6, 12]


def norm_name(s: str) -> str:
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    s = s.lower().replace('-', ' ')
    s = re.sub(r'[^a-z ]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def clean_score(s: str) -> str:
    if pd.isna(s):
        return np.nan
    s = str(s).replace('RET', '').replace('W/O', '').strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def map_pressure(level: str) -> str:
    if level == 'G':
        return 'high'
    if level in {'M', 'F'}:
        return 'medium_high'
    return 'lower'


def load_pbp() -> pd.DataFrame:
    files = [
        PBP_DIR / 'pbp_matches_atp_main_archive.csv',
        PBP_DIR / 'pbp_matches_atp_main_current.csv',
    ]
    frames = [pd.read_csv(f) for f in files]
    pbp = pd.concat(frames, ignore_index=True)
    pbp['date_dt'] = pd.to_datetime(pbp['date'], format='%d %b %y', errors='coerce')
    pbp = pbp[pbp['date_dt'].dt.year.isin([2015, 2023])].copy()
    pbp['year'] = pbp['date_dt'].dt.year
    pbp['p1_n'] = pbp['server1'].map(norm_name)
    pbp['p2_n'] = pbp['server2'].map(norm_name)
    pbp['winner_n'] = np.where(pbp['winner'] == 1, pbp['p1_n'], pbp['p2_n'])
    pbp['loser_n'] = np.where(pbp['winner'] == 1, pbp['p2_n'], pbp['p1_n'])
    pbp['score_n'] = pbp['score'].map(clean_score)
    pbp['match_key_local'] = pbp.index.astype(str)
    return pbp


def load_atp() -> pd.DataFrame:
    frames = [pd.read_csv(ATP_DIR / f'atp_matches_{year}.csv') for year in [2015, 2024]]
    atp = pd.concat(frames, ignore_index=True)
    atp['year'] = atp['tourney_date'] // 10000
    atp['winner_n'] = atp['winner_name'].map(norm_name)
    atp['loser_n'] = atp['loser_name'].map(norm_name)
    atp['score_n'] = atp['score'].map(clean_score)
    atp['pressure_level'] = atp['tourney_level'].map(map_pressure)
    return atp


def merge_datasets(pbp: pd.DataFrame, atp: pd.DataFrame) -> pd.DataFrame:
    merged = pbp.merge(
        atp,
        on=['year', 'winner_n', 'loser_n', 'score_n'],
        how='left',
        suffixes=('_pbp', '_atp'),
        indicator=True,
    )
    matched = merged[merged['_merge'] == 'both'].copy()
    matched['match_id'] = matched['pbp_id'].astype(str)
    matched['rank_gap_abs'] = (matched['winner_rank'] - matched['loser_rank']).abs()
    return matched


def parse_match_points(pbp_str: str, server1_name: str, server2_name: str):
    """
    Returns a list of point dictionaries with:
    point_index, set_no, game_no, point_char, server_name, returner_name,
    point_winner_name, point_winner_role, bp_chance_player, bp_lost_player
    """
    events = []
    current_server = server1_name
    other_player = server2_name
    set_no = 1
    game_no = 0
    point_index = 0

    def is_break_point(server_pts: int, returner_pts: int) -> bool:
        # standard tennis scoring only, tiebreaks excluded before call
        if returner_pts >= 3 and returner_pts > server_pts:
            return True
        return False

    sets = str(pbp_str).split('.')
    for set_chunk in sets:
        if not set_chunk:
            set_no += 1
            continue
        games = [g for g in set_chunk.split(';') if g != '']
        for game in games:
            game_no += 1
            server_pts = 0
            returner_pts = 0
            tiebreak = '/' in game
            if tiebreak:
                # Skip tiebreak internals for break-point logic; no break points in tiebreaks.
                point_chars = [c for c in game if c in 'SRAD']
                for ch in point_chars:
                    point_index += 1
                    point_winner_name = current_server if ch in 'SA' else other_player
                    events.append({
                        'point_index': point_index,
                        'set_no': set_no,
                        'game_no': game_no,
                        'point_char': ch,
                        'server_name': current_server,
                        'returner_name': other_player,
                        'point_winner_name': point_winner_name,
                        'point_winner_role': 'server' if point_winner_name == current_server else 'returner',
                        'bp_chance_player': None,
                        'bp_lost_player': None,
                        'tiebreak': True,
                    })
                current_server, other_player = other_player, current_server
                continue

            for ch in [c for c in game if c in 'SRAD']:
                bp_player = other_player if is_break_point(server_pts, returner_pts) else None
                point_index += 1
                point_winner_name = current_server if ch in 'SA' else other_player
                bp_lost_player = bp_player if (bp_player is not None and point_winner_name == current_server) else None
                events.append({
                    'point_index': point_index,
                    'set_no': set_no,
                    'game_no': game_no,
                    'point_char': ch,
                    'server_name': current_server,
                    'returner_name': other_player,
                    'point_winner_name': point_winner_name,
                    'point_winner_role': 'server' if point_winner_name == current_server else 'returner',
                    'bp_chance_player': bp_player,
                    'bp_lost_player': bp_lost_player,
                    'tiebreak': False,
                })
                if ch in 'SA':
                    server_pts += 1
                else:
                    returner_pts += 1

            current_server, other_player = other_player, current_server
        set_no += 1
    return events


def build_event_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, match in matches.iterrows():
        points = parse_match_points(match['pbp'], match['server1'], match['server2'])
        if not points:
            continue
        total_points = len(points)
        winner_counts = {}
        winners = [pt['point_winner_name'] for pt in points]
        for w in winners:
            winner_counts[w] = winner_counts.get(w, 0) + 1
        bp_events = [pt for pt in points if pt['bp_lost_player'] is not None]
        for pt in bp_events:
            player = pt['bp_lost_player']
            opp = match['server1'] if player == match['server2'] else match['server2']
            idx = pt['point_index']
            row = {
                'match_id': match['match_id'],
                'date': match['date_dt'],
                'tourney_name': match['tourney_name'],
                'surface': match['surface'],
                'tourney_level': match['tourney_level'],
                'pressure_level': match['pressure_level'],
                'round': match['round'],
                'player': player,
                'opponent': opp,
                'set_no': pt['set_no'],
                'game_no': pt['game_no'],
                'point_index': idx,
                'player_rank': match['winner_rank'] if norm_name(player) == match['winner_n'] else match['loser_rank'],
                'opponent_rank': match['loser_rank'] if norm_name(player) == match['winner_n'] else match['winner_rank'],
                'baseline_win_rate': winner_counts.get(player, 0) / total_points,
                'match_winner': match['winner_name'],
            }
            row['rank_diff'] = row['player_rank'] - row['opponent_rank']
            row['is_underdog'] = int(row['rank_diff'] > 0)
            row['won_match'] = int(player == match['winner_name'])
            future_winners = winners[idx: idx + max(K_VALUES)]
            for k in K_VALUES:
                nxt = future_winners[:k]
                if len(nxt) == k:
                    wins = sum(1 for w in nxt if w == player)
                    pbpp = wins / k
                    row[f'pbpp_{k}'] = pbpp
                    row[f'pd_{k}'] = pbpp - row['baseline_win_rate']
                else:
                    row[f'pbpp_{k}'] = np.nan
                    row[f'pd_{k}'] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def hypothesis_tests(events: pd.DataFrame) -> pd.DataFrame:
    results = []
    for k in K_VALUES:
        s = events[f'pd_{k}'].dropna()
        if len(s) > 3:
            tstat, p_two = stats.ttest_1samp(s, 0.0, nan_policy='omit')
            p_one = p_two / 2 if tstat < 0 else 1 - (p_two / 2)
            results.append({
                'test': f'one_sample_ttest_pd_{k}_less_than_zero',
                'n': int(s.shape[0]),
                'mean': float(s.mean()),
                't_stat': float(tstat),
                'p_value_one_sided': float(p_one),
            })
        groups = [g[f'pd_{k}'].dropna().values for _, g in events.groupby('surface') if g[f'pd_{k}'].notna().sum() > 2]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            results.append({
                'test': f'anova_pd_{k}_by_surface',
                'n': int(events[f'pd_{k}'].notna().sum()),
                'mean': float(events[f'pd_{k}'].mean()),
                't_stat': float(f_stat),
                'p_value_one_sided': float(p_val),
            })
    return pd.DataFrame(results)


def save_basic_eda(events: pd.DataFrame):
    summary = {
        'n_events': int(len(events)),
        'n_matches': int(events['match_id'].nunique()),
        'years': sorted(pd.to_datetime(events['date']).dt.year.dropna().unique().tolist()),
        'surface_counts': events['surface'].value_counts(dropna=False).to_dict(),
        'pressure_counts': events['pressure_level'].value_counts(dropna=False).to_dict(),
        'mean_pbpp_3': float(events['pbpp_3'].mean()),
        'mean_pbpp_6': float(events['pbpp_6'].mean()),
        'mean_pbpp_12': float(events['pbpp_12'].mean()),
        'mean_pd_3': float(events['pd_3'].mean()),
        'mean_pd_6': float(events['pd_6'].mean()),
        'mean_pd_12': float(events['pd_12'].mean()),
    }
    pd.Series(summary).to_csv(OUT_DIR / 'eda_summary.csv')


def main():
    pbp = load_pbp()
    atp = load_atp()
    merged = merge_datasets(pbp, atp)
    merged.to_csv(OUT_DIR / 'matched_matches.csv', index=False)
    events = build_event_dataset(merged)
    events.to_csv(OUT_DIR / 'breakpoint_events.csv', index=False)
    tests = hypothesis_tests(events)
    tests.to_csv(OUT_DIR / 'hypothesis_tests.csv', index=False)
    save_basic_eda(events)
    print('Saved outputs to', OUT_DIR)
    print('Matched matches:', merged.shape)
    print('Breakpoint events:', events.shape)
    print(tests)


if __name__ == '__main__':
    main()
