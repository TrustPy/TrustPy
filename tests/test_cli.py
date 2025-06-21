import subprocess
import numpy as np
import sys
from pathlib import Path

def test_trustpy_cli_runs(tmp_path):
    # Create oracle and predictions
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

    # Run without plot
    cmd = [
        sys.executable,
        "-m", "trustpy",
        "--oracle", str(oracle_path),
        "--pred", str(preds_path),
        "--mode", "nts"
    ]

    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Overall" in result.stdout

def test_trustpy_cli_with_trust_spectrum(tmp_path):
    # Data setup
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

    # Run with --trust_spectrum
    cmd = [
        sys.executable,
        "-m", "trustpy",
        "--oracle", str(oracle_path),
        "--pred", str(preds_path),
        "--mode", "nts",
        "--trust_spectrum"
    ]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Overall" in result.stdout

    # Verify plot exists
    expected_plot = repo_root / "trustpy" / "nts" / "trust_spectrum.png"
    assert expected_plot.exists(), f"Plot not found: {expected_plot}"

def test_trustpy_cli_with_custom_output_dir(tmp_path):
    import subprocess
    import sys

    # Arrange: create oracle and preds files
    oracle = np.array([0, 0, 1])
    preds = np.array([
        [0.9, 0.1],
        [1.0, 0.0],
        [0.2, 0.8],
    ])

    oracle_path = tmp_path / "oracle.npy"
    preds_path = tmp_path / "preds.npy"
    output_dir = tmp_path / "custom_output"

    np.save(oracle_path, oracle)
    np.save(preds_path, preds)

    cmd = [
        sys.executable,
        "-m", "trustpy",
        "--oracle", str(oracle_path),
        "--pred", str(preds_path),
        "--mode", "nts",
        "--trust_spectrum",
        "--output_dir", str(output_dir)
    ]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert output_dir.exists(), "Output directory was not created"

    outputs = list(output_dir.glob("*"))
    assert any(f.name.endswith(".png") or f.name.endswith(".csv") for f in outputs), "Expected output files not found"
