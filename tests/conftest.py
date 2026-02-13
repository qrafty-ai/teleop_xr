import os
import shutil
import stat
import sys


def pytest_configure(config):
    if os.name == "nt":
        # Monkeypatch shutil.rmtree to be more robust on Windows
        original_rmtree = shutil.rmtree

        def rmtree_robust_wrapper(path, *args, **kwargs):
            # Try robust version first for simple calls
            if not args and not kwargs:
                try:
                    return rmtree_robust(path)
                except Exception:
                    pass
            # Fall back to original for calls with special arguments
            return original_rmtree(path, *args, **kwargs)

        shutil.rmtree = rmtree_robust_wrapper


def rmtree_robust(path):
    def handle_error(func, path, exc_info):
        # Handle read-only files on Windows
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            # If it's a real permission error or other issue, we can't do much
            # but we shouldn't let it crash the whole test session teardown
            pass

    if os.path.exists(path):
        # Use onexc for Python 3.12+, onerror for older versions
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=handle_error)
        else:
            shutil.rmtree(path, onerror=handle_error)
