import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin, atan2, degrees


# ============================
# Basic geo helpers
# ============================

def _haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in meters between two points given in degrees.
    """
    R = 6371000.0
    phi1, phi2 = map(radians, (lat1, lat2))
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * asin(min(1, sqrt(a)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """
    Forward azimuth A->B in degrees [0, 360).
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dlam = radians(lon2 - lon1)
    x = sin(dlam) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlam)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def _angle_diff_deg(a, b):
    """
    Smallest signed angular difference a->b in [-180, 180].
    """
    return (b - a + 180.0) % 360.0 - 180.0


# ============================
# Direction-based pointwise & summarization (for quality)
# ============================

def _pointwise_dir(traj):
    """
    traj: numpy array with columns [timestamp, lat, lon, alt_FL]
    Returns DataFrame with dt, dist_m, speed_m_s, vertical_rate, bearing, turn, turn rate.
    """
    df = pd.DataFrame(traj, columns=['t', 'lat', 'lon', 'alt_FL']).astype(float)

    # time delta
    df['dt'] = df['t'].diff().fillna(1.0)
    df.loc[df['dt'] == 0, 'dt'] = 1.0  # guard against zero dt

    # horizontal distance and speed
    dists = [0.0]
    for i in range(1, len(df)):
        dists.append(_haversine_m(df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
                                  df.loc[i, 'lat'], df.loc[i, 'lon']))
    df['dist_m'] = dists
    df['speed_m_s'] = df['dist_m'] / df['dt']

    # vertical rate (FL/s)
    df['vertical_rate_FL_s'] = df['alt_FL'].diff() / df['dt']
    df['vertical_rate_FL_s'] = df['vertical_rate_FL_s'].fillna(0.0)

    # bearing A->B
    brngs = [np.nan]
    for i in range(1, len(df)):
        brngs.append(_bearing_deg(df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
                                  df.loc[i, 'lat'], df.loc[i, 'lon']))
    df['bearing_deg'] = brngs

    # turn and turn-rate
    turns = [np.nan]
    for i in range(1, len(df)):
        a = df.loc[i - 1, 'bearing_deg']
        b = df.loc[i, 'bearing_deg']
        turns.append(np.nan if (np.isnan(a) or np.isnan(b)) else _angle_diff_deg(a, b))
    df['turn_deg'] = turns
    df['turn_rate_deg_s'] = df['turn_deg'] / df['dt']

    return df


def _summarize_dir_full(traj):
    """
    Direction-aware summary of a trajectory.
    Used for quality estimation.
    """
    df = _pointwise_dir(traj)

    feats = {}
    # speed stats
    feats['mean_speed_m_s'] = float(df['speed_m_s'].mean())
    feats['std_speed_m_s'] = float(df['speed_m_s'].std())
    feats['max_speed_m_s'] = float(df['speed_m_s'].max())

    # vertical motion
    feats['mean_vrate'] = float(df['vertical_rate_FL_s'].mean())
    feats['max_abs_vrate'] = float(np.abs(df['vertical_rate_FL_s']).max())

    # sinuosity
    total_dist = float(df['dist_m'].sum())
    lat1, lon1 = df.loc[0, 'lat'], df.loc[0, 'lon']
    lat2, lon2 = df.loc[len(df) - 1, 'lat'], df.loc[len(df) - 1, 'lon']
    crow = _haversine_m(lat1, lon1, lat2, lon2)
    feats['sinuosity'] = float(total_dist / crow) if crow > 0 else 1.0

    # turning behaviour
    turn = df['turn_deg'].dropna()
    trate = df['turn_rate_deg_s'].dropna()

    feats['mean_abs_turn_deg'] = float(np.abs(turn).mean() if len(turn) else 0.0)
    feats['max_abs_turn_deg'] = float(np.abs(turn).max() if len(turn) else 0.0)
    feats['mean_abs_turn_rate_deg_s'] = float(np.abs(trate).mean() if len(trate) else 0.0)

    return feats


# ============================
# Basic summarization (for stats-based optimizer thresholds)
# ============================

def _summarize_basic(traj):
    """
    Simpler summary akin to your original summarize().
    traj: numpy array [timestamp, lat, lon, alt_FL]
    """
    df = pd.DataFrame(traj, columns=['t', 'lat', 'lon', 'alt_FL']).astype(float)

    # time delta
    df['dt'] = df['t'].diff().fillna(1.0)
    df['dt'] = df['dt'].replace(0, np.nan)

    # distances
    dists = [0.0]
    for i in range(1, len(df)):
        dists.append(_haversine_m(df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
                                  df.loc[i, 'lat'], df.loc[i, 'lon']))
    df['dist_m'] = dists

    # speeds
    df['speed_m_s'] = df['dist_m'] / df['dt']
    df['speed_m_s'] = df['speed_m_s'].fillna(0.0)

    # vertical rate
    df['vertical_rate_FL_s'] = df['alt_FL'].diff() / df['dt']
    df['vertical_rate_FL_s'] = df['vertical_rate_FL_s'].fillna(0.0)

    feats = {}
    feats['mean_speed_m_s'] = float(df['speed_m_s'].mean())
    feats['std_speed_m_s'] = float(df['speed_m_s'].std())
    feats['max_speed_m_s'] = float(df['speed_m_s'].max())
    feats['mean_vrate'] = float(df['vertical_rate_FL_s'].mean())
    feats['max_abs_vrate'] = float(np.abs(df['vertical_rate_FL_s']).max())
    feats['total_dist_m'] = float(df['dist_m'].sum())

    # sinuosity
    lat1, lon1 = df.loc[0, 'lat'], df.loc[0, 'lon']
    lat2, lon2 = df.loc[len(df) - 1, 'lat'], df.loc[len(df) - 1, 'lon']
    crow = _haversine_m(lat1, lon1, lat2, lon2)
    feats['sinuosity'] = feats['total_dist_m'] / crow if crow > 0 else 1.0

    feats['frac_zero_speed'] = float((df['speed_m_s'] == 0).mean())
    feats['n_large_alt_jumps'] = int((df['alt_FL'].diff().abs() > 50).sum())

    return feats


# ============================
# Stats-based quality (requires GOOD trajectories)
# ============================

def _compute_quality_stats_good(good_trajectories):
    """
    Compute mu, std and feature list from GOOD trajectories for quality scoring.
    """
    feat_list = []
    for traj in good_trajectories:
        try:
            feat_list.append(_summarize_dir_full(traj))
        except Exception:
            continue

    df_good = pd.DataFrame(feat_list)
    df_good = df_good.replace([np.inf, -np.inf], np.nan).dropna()

    mu_good = df_good.mean(numeric_only=True)
    std_good = df_good.std(numeric_only=True).replace(0, 1e-9)
    features = list(df_good.columns)
    return mu_good, std_good, features


def _quality_from_features_stats(feats, mu_good, std_good, feature_names):
    """
    Dataset-based quality:
    - z-score distance from GOOD cluster
    - values within 2 std -> no penalty
    """
    penalties = []
    for feat in feature_names:
        x = feats[feat]
        mu = mu_good[feat]
        sd = std_good[feat]
        z = abs(x - mu) / sd
        excess = max(0, z - 2.0)
        penalty = min(excess / 3.0, 1.0)
        penalties.append(penalty)

    if not penalties:
        return 100.0

    avg_penalty = float(np.mean(penalties))
    quality = (1.0 - avg_penalty) * 100.0
    return round(float(np.clip(quality, 0.0, 100.0)), 2)


# ============================
# Universal aggressive quality (fixed ranges)
# ============================

UNIVERSAL_NORMAL_RANGES = {
    'mean_speed_m_s':           (80.0, 280.0),
    'std_speed_m_s':            (0.0,  80.0),
    'max_speed_m_s':            (100.0, 320.0),
    'mean_vrate':               (-10.0, 10.0),
    'max_abs_vrate':            (0.0,  25.0),
    'sinuosity':                (1.0,  2.5),
    'mean_abs_turn_deg':        (0.0,  25.0),
    'max_abs_turn_deg':         (0.0,  120.0),
    'mean_abs_turn_rate_deg_s': (0.0,  4.0),
}


def _quality_from_features_universal(feats):
    """
    Universal aggressive quality based on fixed normal ranges.
    """
    penalties = []

    for feat, (low, high) in UNIVERSAL_NORMAL_RANGES.items():
        x = feats[feat]

        if low <= x <= high:
            penalties.append(0.0)
            continue

        if x < low:
            diff = low - x
            span = max(low, 1e-6)
        else:
            diff = x - high
            span = max(high - low, 1e-6)

        rel = diff / span
        penalty = min(rel, 1.0)
        penalties.append(penalty)

    if not penalties:
        return 100.0

    avg_penalty = float(np.mean(penalties))
    quality = (1.0 - avg_penalty) * 100.0
    return round(float(np.clip(quality, 0.0, 100.0)), 2)


# ============================
# Internal: helpers for optimization
# ============================

def _remove_velocity_outliers(df, max_speed):
    """
    Remove points where the segment speed exceeds max_speed (m/s).
    df must have columns: ['timestamp', 'lat', 'lon', 'alt_FL'].
    """
    if len(df) < 2:
        return df

    keep_mask = [True]  # first point always kept

    for i in range(1, len(df)):
        dist = _haversine_m(
            df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'],
            df.loc[i, 'lat'], df.loc[i, 'lon']
        )
        time_diff = df.loc[i, 'timestamp'] - df.loc[i - 1, 'timestamp']
        speed = dist / time_diff if time_diff > 0 else 0.0
        keep_mask.append(speed <= max_speed)

    keep_mask = np.array(keep_mask)
    return df[keep_mask].reset_index(drop=True)


def _optimize_sinuosity(df, max_sinuosity):
    """
    Remove points that cause strong backtracking to reduce path sinuosity.
    """
    if len(df) < 4:
        return df

    # Direct distance start→end
    direct_dist = _haversine_m(
        df.iloc[0]['lat'], df.iloc[0]['lon'],
        df.iloc[-1]['lat'], df.iloc[-1]['lon']
    )
    if direct_dist == 0:
        return df

    # Current path length
    total_dist = 0.0
    for i in range(1, len(df)):
        total_dist += _haversine_m(
            df.iloc[i - 1]['lat'], df.iloc[i - 1]['lon'],
            df.iloc[i]['lat'], df.iloc[i]['lon']
        )

    current_sinuosity = total_dist / direct_dist if direct_dist > 0 else 1.0
    if current_sinuosity <= max_sinuosity:
        return df

    # Angle-based pruning of backtracking points
    keep = [True] * len(df)
    keep[0] = True
    keep[-1] = True

    for i in range(1, len(df) - 1):
        v1_lat = df.iloc[i]['lat'] - df.iloc[i - 1]['lat']
        v1_lon = df.iloc[i]['lon'] - df.iloc[i - 1]['lon']
        v2_lat = df.iloc[i + 1]['lat'] - df.iloc[i]['lat']
        v2_lon = df.iloc[i + 1]['lon'] - df.iloc[i]['lon']

        mag1 = np.sqrt(v1_lat ** 2 + v1_lon ** 2)
        mag2 = np.sqrt(v2_lat ** 2 + v2_lon ** 2)

        if mag1 == 0 or mag2 == 0:
            continue

        dot = v1_lat * v2_lat + v1_lon * v2_lon
        cos_angle = dot / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)  # radians

        # >135° = “backtracking / sharp U-turn”
        if angle > np.radians(135):
            keep[i] = False

    df_opt = df[keep].reset_index(drop=True)
    return df_opt


def _compute_basic_good_stats(good_trajectories):
    """
    Compute mean feature statistics for GOOD trajectories (basic features).
    Used for stats-based optimization thresholds.
    """
    feat_list = []
    for traj in good_trajectories:
        try:
            feat_list.append(_summarize_basic(traj))
        except Exception:
            continue

    df_good = pd.DataFrame(feat_list)
    real_good_stats = df_good.mean().to_dict()
    return real_good_stats


# ============================
# PUBLIC FUNCTIONS
# ============================

def estimate_quality_stats(traj, good_trajectories):
    """
    Dataset-based quality estimate in [0, 100] compared to GOOD trajectories.

    Parameters
    ----------
    traj : np.ndarray
        Shape (N, 4): [timestamp, lat, lon, alt_FL]
    good_trajectories : list of np.ndarray
        List of GOOD trajectories (same format).

    Returns
    -------
    float
        Quality in percent (0–100).
    """
    mu_good, std_good, feature_names = _compute_quality_stats_good(good_trajectories)
    feats = _summarize_dir_full(traj)
    return _quality_from_features_stats(feats, mu_good, std_good, feature_names)


def estimate_quality_universal(traj):
    """
    Universal aggressive quality estimate in [0, 100] using fixed
    normal ranges for direction-based features (no dataset needed).

    Parameters
    ----------
    traj : np.ndarray
        Shape (N, 4): [timestamp, lat, lon, alt_FL]

    Returns
    -------
    float
        Quality in percent (0–100).
    """
    feats = _summarize_dir_full(traj)
    return _quality_from_features_universal(feats)


def optimize_trajectory_aggressive(trajectory):
    """
    Aggressive optimization with FIXED thresholds:
      - Speed limit: 300 m/s
      - Sinuosity limit: 1.5

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (N, 4): [timestamp, lat, lon, alt_FL]

    Returns
    -------
    np.ndarray
        Optimized trajectory with same 4 columns.
    """
    df = pd.DataFrame(trajectory, columns=['timestamp', 'lat', 'lon', 'alt_FL'])

    # Sort + deduplicate by timestamp
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # Remove crazy GPS spikes (very high speed)
    df = _remove_velocity_outliers(df, max_speed=300.0)

    # Reduce path backtracking / curvature
    if len(df) > 8:
        df = _optimize_sinuosity(df, max_sinuosity=1.5)

    return df.values


def optimize_trajectory_stats_based(trajectory, good_trajectories):
    """
    Optimization using thresholds derived from GOOD trajectories:
      - Speed limit: mean_good_max_speed * 1.1
      - Sinuosity limit: mean_good_sinuosity * 1.2

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (N, 4): [timestamp, lat, lon, alt_FL]
    good_trajectories : list of np.ndarray
        List of GOOD trajectories (used to compute thresholds).

    Returns
    -------
    np.ndarray
        Optimized trajectory with same 4 columns.
    """
    df = pd.DataFrame(trajectory, columns=['timestamp', 'lat', 'lon', 'alt_FL'])

    # Sort + deduplicate by timestamp
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    # Thresholds from good trajectories
    good_stats = _compute_basic_good_stats(good_trajectories)
    max_speed = good_stats['max_speed_m_s'] * 1.1       # +10% buffer
    max_sinuosity = good_stats['sinuosity'] * 1.2       # +20% buffer

    # Speed-based cleaning
    df = _remove_velocity_outliers(df, max_speed=max_speed)

    # Sinuosity-based cleaning
    if len(df) > 8:
        df = _optimize_sinuosity(df, max_sinuosity=max_sinuosity)

    return df.values


__all__ = [
    "estimate_quality_stats",
    "estimate_quality_universal",
    "optimize_trajectory_aggressive",
    "optimize_trajectory_stats_based",
]
