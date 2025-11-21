import os
import pickle
import tempfile
import cloudpickle
import joblib


def _atomic_write(path, writer):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try: os.remove(tmp)
        except OSError: pass


def joblib_cloud_atomic_dump(obj, path, *, compress=3, protocol=pickle.HIGHEST_PROTOCOL):
    blob = cloudpickle.dumps(obj, protocol=protocol)
    _atomic_write(path, lambda tmp: joblib.dump(blob, tmp, compress=compress))


def joblib_cloud_load(path):
    blob = joblib.load(path)
    return cloudpickle.loads(blob)


def joblib_atomic_dump(obj, path, *, compress=3, protocol=None):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        joblib.dump(obj, tmp, compress=compress, protocol=protocol)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass


def joblib_safe_load(path, *, mmap_mode=None):
    # Try a strict load first; if it fails with EOF, surface a clear message.
    try:
        return joblib.load(path, mmap_mode=mmap_mode)
    except EOFError as e:
        raise EOFError(f"{path} appears truncated/corrupted. "
                       "Recreate it with an atomic dump and avoid concurrent writers.") from e
