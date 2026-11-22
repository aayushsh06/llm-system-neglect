"""Load the Experiment 1 human data from the OSF archive (project xh7dy).

The subject tables are MATLAB ``table`` objects, which scipy cannot read
directly; :func:`decode_tables` reads them out of the file's MCOS subsystem
stream. Nothing here reconstructs or imputes human data.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

from .task import D_LEVELS, N_PERIODS, Q_LEVELS, Trial, condition_index, d_level, q_level

OSF_PROJECT_URL = "https://osf.io/xh7dy/"


@dataclass(frozen=True)
class OsfFile:
    name: str
    url: str
    md5: str
    archive_path: str


OSF_FILES: dict[str, OsfFile] = {
    "data_allsubj_expt1.mat": OsfFile(
        "data_allsubj_expt1.mat",
        "https://osf.io/download/kas93/",
        "9052e8a0a8748573ffb6308e4ab729bd",
        "1.Raw_data/behavior/data_allsubj_expt1.mat",
    ),
    "BETA_neglect_final.mat": OsfFile(
        "BETA_neglect_final.mat",
        "https://osf.io/download/660a74e4943bee089edfec1e/",
        "d074561374c374162914b8283b8ec92c",
        "2.Analysis/parameter/BETA_neglect_final.mat",
    ),
    "BETA_power_final.mat": OsfFile(
        "BETA_power_final.mat",
        "https://osf.io/download/660a74e5e65c6008cc7d9bd4/",
        "8bd4f12a20a8252d9538e0088dcbeffb",
        "2.Analysis/parameter/BETA_power_final.mat",
    ),
}

# Published means (eLife 104684, Results; equal to the column means of BETA_neglect_final.mat).
PUBLISHED_ALPHA_MEANS: tuple[float, float, float] = (3.69, 1.04, 0.65)
PUBLISHED_BETA_MEANS: tuple[float, float, float] = (1.69, 0.77, 0.57)
HUMAN_SOURCE = "human"


def md5_of(path: Path) -> str:
    """md5 hex digest of a file, for comparison with the checksum recorded on OSF."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - integrity check only


def download_human_data(dest_dir: str | Path, timeout: float = 120.0) -> dict[str, Path]:
    """Download the data and parameter files from OSF into ``dest_dir`` and verify md5s.

    Files already present with the right hash are not downloaded again.
    Raises ``RuntimeError`` if a download fails or a hash does not match; the
    caller must stop rather than fall back to synthetic data.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, spec in OSF_FILES.items():
        target = dest / name
        if target.exists() and md5_of(target) == spec.md5:
            out[name] = target
            continue
        try:
            with urllib.request.urlopen(spec.url, timeout=timeout) as r:  # noqa: S310
                data = r.read()
        except Exception as exc:  # pragma: no cover - network
            raise RuntimeError(f"download of {spec.url} failed: {exc}") from exc
        got = hashlib.md5(data).hexdigest()  # noqa: S324
        if got != spec.md5:
            raise RuntimeError(f"{name}: md5 {got} does not match OSF's {spec.md5}")
        target.write_bytes(data)
        out[name] = target
    return out


def _object_metadata(mcos: object) -> np.ndarray:
    """Find the FileWrapper__ cell array inside the decoded MCOS variable.

    scipy names the fields of an opaque object differently across versions
    (``_ObjectMetadata`` in recent releases, ``arr`` in older ones), so the
    cell array is located by structure: the only object-dtype column vector
    with more than a handful of cells.
    """
    seen: list[object] = []

    def walk(obj: object, depth: int = 0) -> np.ndarray | None:
        if depth > 6 or any(obj is s for s in seen):
            return None
        seen.append(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype == object and obj.ndim == 2 and obj.shape[1] == 1 and obj.shape[0] > 8:
                first = obj[0, 0]
                if isinstance(first, np.ndarray) and first.dtype == np.uint8:
                    return obj
            if obj.dtype.names:
                for name in obj.dtype.names:
                    for item in np.asarray(obj[name]).ravel():
                        found = walk(item, depth + 1)
                        if found is not None:
                            return found
            elif obj.dtype == object:
                for item in obj.ravel():
                    found = walk(item, depth + 1)
                    if found is not None:
                        return found
        return None

    found = walk(mcos)
    if found is None:
        raise ValueError(
            "could not locate the MATLAB table contents in the MCOS subsystem "
            f"(scipy {__import__('scipy').__version__}); the file may not be the OSF archive"
        )
    return found


def decode_tables(path: str | Path) -> list[pd.DataFrame]:
    """Decode the cell array of MATLAB tables in ``data_allsubj_expt*.mat``.

    MATLAB stores ``table`` objects as opaque MCOS objects whose contents live
    in the ``__function_workspace__`` stream. Each table occupies seven
    consecutive cells there: column data, ndims, nrows, row names, nvars,
    variable names, properties.
    """
    m = sio.loadmat(str(path))
    if "__function_workspace__" not in m:
        raise ValueError(f"{path} has no MCOS subsystem; is it the right file?")
    buf = m["__function_workspace__"].tobytes()
    reader = MatFile5Reader(io.BytesIO(buf))
    reader.byte_order = "<"
    reader.mat_stream.seek(8)  # 8-byte subsystem header instead of the 128-byte file header
    reader.initialize_read()
    header, _ = reader.read_var_header()
    mcos = reader.read_var_array(header, True)
    metadata = _object_metadata(mcos)
    tables: list[pd.DataFrame] = []
    i = 2
    while i + 6 < metadata.shape[0]:
        data, _ndims, _nrows, _rownames, nvars, varnames, _props = (
            metadata[i + k, 0] for k in range(7)
        )
        if not (
            isinstance(nvars, np.ndarray)
            and nvars.size == 1
            and isinstance(varnames, np.ndarray)
            and varnames.dtype == object
        ):
            break
        names = [str(v[0]) for v in varnames[0]]
        cols: dict[str, object] = {}
        for name, col in zip(names, data[0], strict=True):
            arr = np.asarray(col)
            if arr.ndim == 2 and arr.shape[1] == 1 or arr.ndim == 1:
                cols[name] = arr.reshape(-1)
            else:
                cols[name] = list(arr)
        tables.append(pd.DataFrame(cols))
        i += 7
    return tables


def load_human_long(path: str | Path, trial_set: str = "all") -> pd.DataFrame:
    """Experiment 1 as one long table: one row per (subject, trial, period).

    Columns: ``source`` (always ``"human"``), ``subject``, ``session``, ``block``,
    ``trial``, ``trial_in_block``, ``condition``, ``q``, ``d``, ``period``, ``signal``,
    ``estimate`` (NaN when no response was recorded), ``rt`` and ``p_bayes_archive``
    (the archive's own Bayesian posterior).

    ``trial_set``: ``"all"`` (99 trials per subject, what the published fits
    used) or ``"behavioral"`` (72 trials from the behavioural session only).
    """
    if trial_set not in ("all", "behavioral"):
        raise ValueError("trial_set must be 'all' or 'behavioral'")
    frames = []
    for tbl in decode_tables(path):
        if len(tbl) % N_PERIODS:
            raise ValueError("subject table length is not a multiple of 10")
        obs = tbl["obsNo"].to_numpy()
        if not (obs.reshape(-1, N_PERIODS) == np.arange(1, N_PERIODS + 1)).all():
            raise ValueError("periods are not contiguous 1..10 blocks")
        est = tbl["probability_estimate"].to_numpy(dtype=float).copy()
        rt = tbl["RT"].to_numpy(dtype=float)
        est[rt == -1] = np.nan
        odds = tbl["Qt_odds"].to_numpy(dtype=float) * tbl["Dt"].to_numpy(dtype=float)
        n_trials = len(tbl) // N_PERIODS
        trial_id = np.repeat(np.arange(1, n_trials + 1), N_PERIODS)
        # The archive stores d as a ratio of color proportions (0.9/0.1 = 9.000000000000002);
        # snap q and d to the design levels so they can be used as keys.
        q = np.array([Q_LEVELS[q_level(x)] for x in tbl["Q"].to_numpy(dtype=float)])
        d = np.array([D_LEVELS[d_level(x)] for x in tbl["D"].to_numpy(dtype=float)])
        frames.append(
            pd.DataFrame(
                {
                    "source": HUMAN_SOURCE,
                    "subject": tbl["subNO"].to_numpy(dtype=int),
                    "session": tbl["fMRI_session"].to_numpy(dtype=int),
                    "block": tbl["blockNo"].to_numpy(dtype=int),
                    "trial": trial_id,
                    "trial_in_block": tbl["trialNo"].to_numpy(dtype=int),
                    "condition": tbl["index_condition"].to_numpy(dtype=int),
                    "q": q,
                    "d": d,
                    "period": obs.astype(int),
                    "signal": tbl["signal"].to_numpy(dtype=int),
                    "estimate": est,
                    "rt": rt,
                    "p_bayes_archive": odds / (1.0 + odds),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    check = np.array([condition_index(a, b) for a, b in zip(df.q, df.d, strict=True)])
    if not (check == df.condition.to_numpy()).all():
        raise ValueError("archive condition index does not match (q, d)")
    if trial_set == "behavioral":
        df = df[df.session == 0].reset_index(drop=True)
    return df


def human_trials(long_df: pd.DataFrame) -> dict[int, list[Trial]]:
    """Each subject's trials as :class:`Trial` objects (change points unknown)."""
    out: dict[int, list[Trial]] = {}
    key = ["subject", "trial"]
    for (subject, trial), g in long_df.sort_values([*key, "period"]).groupby(key, sort=True):
        row = g.iloc[0]
        out.setdefault(int(subject), []).append(
            Trial(
                subject=int(subject),
                trial=int(trial),
                session=int(row.session),
                block=int(row.block),
                trial_in_block=int(row.trial_in_block),
                q=float(row.q),
                d=float(row.d),
                signals=tuple(int(s) for s in g.signal),
                change_period=None,
            )
        )
    return out


def load_published_parameters(path: str | Path) -> pd.DataFrame:
    """Archived per-subject estimates: columns alpha_1..3 (q levels), beta_1..3 (d levels)."""
    arr = sio.loadmat(str(path))["BETA_neg"]
    df = pd.DataFrame(arr, columns=["alpha_1", "alpha_2", "alpha_3", "beta_1", "beta_2", "beta_3"])
    df.insert(0, "subject", np.arange(1, len(df) + 1))
    return df
