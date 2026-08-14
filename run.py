"""
MediSynth.AI — One-Command Startup
"""
import subprocess
import sys
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  MediSynth.AI -- Starting Up")
    print("  Privacy-Preserving Synthetic Healthcare Data Platform")
    print("=" * 60 + "\n")

    # Install dependencies
    print("[*] Checking dependencies...")
    try:
        import fastapi, uvicorn, pandas, numpy, scipy, sklearn
        print("    All core dependencies found")
    except ImportError:
        print("    Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "fastapi", "uvicorn[standard]", "python-multipart",
            "aiofiles", "pandas", "numpy", "scipy", "scikit-learn",
            "pydantic",
        ])
        print("    Dependencies installed")

    # Generate sample data
    from backend.config import DATA_DIR
    sample_path = DATA_DIR / "sample" / "healthcare_data.csv"
    if not sample_path.exists():
        print("[*] Generating sample healthcare dataset...")
        _generate_sample(sample_path)
        print(f"    Created {sample_path}")

    print(f"\n[*] Starting server at http://127.0.0.1:8000")
    print(f"[*] API docs at http://127.0.0.1:8000/docs")
    print(f"[*] Dashboard at http://127.0.0.1:8000\n")

    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


def _generate_sample(path):
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "age": np.random.normal(55, 15, n).clip(18, 95).astype(int),
        "gender": np.random.choice(["Male", "Female"], n, p=[0.48, 0.52]),
        "blood_pressure_systolic": np.random.normal(130, 20, n).clip(80, 200).astype(int),
        "blood_pressure_diastolic": np.random.normal(82, 12, n).clip(50, 130).astype(int),
        "cholesterol": np.random.normal(200, 40, n).clip(100, 400).astype(int),
        "bmi": np.random.normal(27, 5, n).clip(15, 50).round(1),
        "glucose": np.random.normal(110, 30, n).clip(60, 300).astype(int),
        "heart_rate": np.random.normal(75, 12, n).clip(45, 130).astype(int),
        "smoking_status": np.random.choice(["Never", "Former", "Current"], n, p=[0.45, 0.30, 0.25]),
    })
    risk = (
        (data["age"] > 55).astype(float) * 0.3 +
        (data["bmi"] > 30).astype(float) * 0.2 +
        (data["cholesterol"] > 240).astype(float) * 0.15 +
        (data["blood_pressure_systolic"] > 140).astype(float) * 0.2 +
        (data["glucose"] > 140).astype(float) * 0.15 +
        (data["smoking_status"] == "Current").astype(float) * 0.2
    )
    data["diabetes"] = (risk + np.random.normal(0, 0.15, n) > 0.45).astype(int)
    data["hypertension"] = ((data["blood_pressure_systolic"] > 140).astype(int) | (risk + np.random.normal(0, 0.1, n) > 0.5).astype(int))
    data["heart_disease"] = (risk + np.random.normal(0, 0.2, n) > 0.55).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


if __name__ == "__main__":
    main()
