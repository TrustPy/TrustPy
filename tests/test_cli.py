import subprocess
import numpy as np
import sys
import os
from pathlib import Path

def test_trustpy_cli_runs(tmp_path):
    # Step 1: Create oracle and prediction files
    oracle = np.array([0, 0, 1])
    preds = np.array([
        [0.9, 0.1],
        [1.0, 0.0],
        [0.2, 0.8],
    ])

    oracle_path = tmp_path / "oracle.npy"
    preds_path = tmp_path / "preds.npy"
    np.save(oracle_path, oracle)
    np.save(preds_path, preds)

    # Step 2: Build command
    cmd = [
        sys.executable,  # the current Python interpreter
        "-m", "trustpy",
        "--oracle", str(oracle_path),
        "--pred", str(preds_path),
        "--mode", "nts"
    ]

    # Step 3: Set working directory to project root (where trustpy/ lives)
    project_root = Path(__file__).resolve().parents[1]

    # Step 4: Run subprocess and capture output
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Step 5: Assertions
    assert result.returncode == 0, f"CLI exited with code {result.returncode}"
    assert "Class" in result.stdout, "Expected output summary not found"
    assert "Overall" in result.stdout, "Missing overall NTS output"

