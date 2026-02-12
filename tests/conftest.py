import os
import shutil
import stat


def pytest_configure(config):
    if os.name == "nt":
        # Monkeypatch shutil.rmtree to be more robust on Windows
        original_rmtree = shutil.rmtree

        def rmtree_robust_wrapper(path, ignore_errors=False, onerror=None):
            if onerror is None:
                return rmtree_robust(path)
            return original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)

        shutil.rmtree = rmtree_robust_wrapper


def rmtree_robust(path):
    def on_error(func, path, exc_info):
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            if os.path.exists(path):
                raise

    if os.path.exists(path):
        shutil.rmtree(path, onerror=on_error)
