"""
logging_utils.py
----------------
CSV logging helpers for MediON runtime. Keeps per-day (or per-session) log
files with a fixed header and provides timestamp formatting helpers.
"""

import time

try:
    import uos as os  # type: ignore
except ImportError:  # pragma: no cover
    import os  # type: ignore


LOG_DIR = "logs"
LOG_HEADERS = [
    "timestamp_start_iso",
    "timestamp_end_iso",
    "hr_mean",
    "hr_std",
    "rr_rmssd",
    "acdc_ratio",
    "ppg_sqi",
    "beat_count",
    "acc_var",
    "motion_frac",
    "hour_sin",
    "hour_cos",
    "hr_z",
    "rr_rmssd_z",
    "label",
    "label_source",
    "dropped_by_qc",
    "stress_prob",
    "alert_fired",
]


def _rtc_has_date():
    try:
        year = time.localtime()[0]
        return year >= 2020
    except Exception:
        return False


def ensure_logs_dir():
    try:
        os.mkdir(LOG_DIR)
    except OSError:
        pass
    return LOG_DIR


def _listdir(path):
    try:
        return os.listdir(path)
    except OSError:
        return []


def _path_join(*parts):
    return "/".join(part.strip("/") for part in parts if part)


def _file_exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def _next_session_filename():
    existing = _listdir(LOG_DIR)
    max_idx = 0
    for name in existing:
        if not name.startswith("session_") or not name.endswith(".csv"):
            continue
        num_part = name[8:-4]
        try:
            idx = int(num_part)
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return "session_{:04d}.csv".format(max_idx + 1)


def _current_date_filename():
    t = time.localtime()
    return "{:04d}-{:02d}-{:02d}.csv".format(t[0], t[1], t[2])


def isoformat_from_epoch(epoch_seconds):
    try:
        tm = time.localtime(epoch_seconds)
        return "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(
            tm[0], tm[1], tm[2], tm[3], tm[4], tm[5]
        )
    except Exception:
        return ""


def isoformat_from_uptime(uptime_seconds):
    return "T+{:.3f}s".format(uptime_seconds)


class CSVLogger:
    """
    Manages rolling CSV logs. Automatically creates the correct filename
    based on RTC availability and writes headers when needed.
    """

    def __init__(self):
        ensure_logs_dir()
        self.current_path = None
        self._rtc_mode = _rtc_has_date()
        self._ensure_path()

    def _ensure_path(self):
        if self._rtc_mode != _rtc_has_date():
            self._rtc_mode = _rtc_has_date()
            self.current_path = None

        if self.current_path and _file_exists(self.current_path):
            if self._rtc_mode:
                # Rotate daily if filename no longer matches today.
                expected = _path_join(LOG_DIR, _current_date_filename())
                if self.current_path != expected:
                    self.current_path = None
            return

        if self._rtc_mode:
            filename = _current_date_filename()
        else:
            filename = _next_session_filename()
        self.current_path = _path_join(LOG_DIR, filename)

    def append(self, row_values):
        """
        Append a CSV row (iterable of strings). Automatically writes the header
        if the file is new.
        """
        self._ensure_path()
        if self.current_path is None:
            return None

        new_file = not _file_exists(self.current_path)
        try:
            with open(self.current_path, "a") as fp:
                if new_file:
                    fp.write(",".join(LOG_HEADERS) + "\n")
                fp.write(",".join(row_values) + "\n")
        except OSError:
            return None
        return self.current_path


__all__ = [
    "CSVLogger",
    "LOG_HEADERS",
    "ensure_logs_dir",
    "isoformat_from_epoch",
    "isoformat_from_uptime",
]
