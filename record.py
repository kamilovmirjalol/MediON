from utime import ticks_ms, sleep
import ujson as json

# record.py - simple MicroPython recorder for HR and HRV (newline-delimited JSON)

RECORD_PATH = "record.txt"

def append_record(hr, hrv):
    """
    Append a single sample to RECORD_PATH as a JSON object with a timestamp (ms).
    hr and hrv may be None.
    """
    rec = {
        "ts": int(ticks_ms()),
        "hr": int(hr) if hr is not None else None,
        "hrv": int(hrv) if hrv is not None else None
    }
    try:
        f = open(RECORD_PATH, "a")
        f.write(json.dumps(rec) + "\n")
        f.close()
    except Exception as e:
        # In MicroPython print to console if file write fails
        print("record append error:", e)

def read_records():
    """
    Read all records from RECORD_PATH and return as a list of dicts.
    Ignores malformed lines.
    """
    out = []
    try:
        f = open(RECORD_PATH, "r")
    except Exception:
        return out
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # skip bad lines
            pass
    f.close()
    return out

# Minimal example usage:
# Call append_record(current_hr, current_hrv) from your main loop (where you have
# current_hr/current_hrv variables). Example standalone demo below.

if __name__ == "__main__":
    # demo: write a few sample entries (remove in production)
    append_record(72, 58)
    sleep(0.1)
    append_record(71, 60)
    print("Saved records:", read_records())