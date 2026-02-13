# Issues - Windows Path Compatibility

- `pytest-cov` was requested in `pyproject.toml` but not installed in the
  environment, causing `pytest` to fail with unrecognized argument `--cov`.
  Overridden using `-o "addopts="`.
- `pytest-asyncio` was not installed, so I switched to using `anyio` which was
  already available as a pytest plugin in the environment.
