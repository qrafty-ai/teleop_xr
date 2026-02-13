from pathlib import Path


def test_ci_workflow_windows_logic():
    """
    Ensure the CI workflow doesn't use the buggy falsy check for Windows.
    Buggy: matrix.os == 'windows-latest' && ''
    Robust: matrix.os != 'windows-latest'
    """
    workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    assert workflow_path.exists(), f"Workflow file not found at {workflow_path}"

    content = workflow_path.read_text()

    # 1. Assert buggy version is NOT present
    # The buggy version was: matrix.os == 'windows-latest' && ''
    # which fails in GitHub Actions because '' is falsy.
    buggy_str = "matrix.os == 'windows-latest' && ''"
    assert buggy_str not in content, f"Buggy string '{buggy_str}' found in CI workflow!"

    # 2. Assert robust version IS present
    robust_str = "matrix.os != 'windows-latest'"
    assert robust_str in content, (
        f"Robust string '{robust_str}' not found in CI workflow!"
    )
