"""End-to-end test for MediSynth.AI — all features."""
import requests
import json
import sys
import time

BASE = "http://127.0.0.1:8000/api"
PASS_COUNT = 0
FAIL_COUNT = 0

def test(name, fn):
    global PASS_COUNT, FAIL_COUNT
    try:
        result = fn()
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
        return result
    except Exception as e:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}: {e}")
        return None

print("\n=== MediSynth.AI E2E Test ===\n")

# 1. Health
test("Health check", lambda: (
    requests.get(f"{BASE}/health").json()
))

# 2. Load sample
print("\n--- Data ---")
sample_res = test("Load sample data", lambda: (
    requests.get(f"{BASE}/data/sample").json()
))
ds_id = sample_res["data"]["dataset_id"] if sample_res else None
if ds_id:
    print(f"    Dataset ID: {ds_id}, Rows: {sample_res['data']['num_rows']}")

# 3. Generate
print("\n--- Generation ---")
gen_res = None
if ds_id:
    gen_res = test("Generate synthetic data (statistical + DP)", lambda: (
        requests.post(f"{BASE}/generate", json={
            "dataset_id": ds_id, "num_rows": 500, "model_type": "statistical",
            "epsilon": 1.0, "delta": 1e-5, "apply_dp": True
        }).json()
    ))
    if gen_res and gen_res.get("data"):
        d = gen_res["data"]
        print(f"    Rows: {d['num_rows_generated']}, Model: {d['model_type']}, DP: {d['dp_applied']}")
        if d.get("dp_metadata"):
            print(f"    Epsilon actual: {d['dp_metadata']['epsilon_actual']}")
        job_id = d["job_id"]
    else:
        job_id = None
else:
    job_id = None

# 4. Privacy budget
print("\n--- Privacy Budget ---")
if ds_id:
    budget_res = test("Privacy budget check", lambda: (
        requests.get(f"{BASE}/privacy/budget/{ds_id}").json()
    ))
    if budget_res and budget_res.get("data"):
        b = budget_res["data"]
        print(f"    Used: {b['total_epsilon_used']}, Remaining: {b['remaining_epsilon']}, Utilization: {b['utilization_pct']}%")

# 5. Statistical validation
print("\n--- Statistical Validation ---")
if ds_id and job_id:
    stat_res = test("Statistical validation", lambda: (
        requests.post(f"{BASE}/validate/statistical", json={
            "dataset_id": ds_id, "synthetic_job_id": job_id
        }).json()
    ))
    if stat_res and stat_res.get("data"):
        d = stat_res["data"]
        print(f"    Quality: {d['overall_quality_score']}/100, Grade: {d['quality_grade']}")
        print(f"    Correlation MAE: {d['correlation']['mean_absolute_error']}")

# 6. ML validation
print("\n--- ML Utility Validation ---")
if ds_id and job_id:
    ml_res = test("ML utility validation (TSTR)", lambda: (
        requests.post(f"{BASE}/validate/ml", json={
            "dataset_id": ds_id, "synthetic_job_id": job_id
        }).json()
    ))
    if ml_res and ml_res.get("data"):
        d = ml_res["data"]
        print(f"    Utility: {d['utility_score']}/100, Grade: {d['utility_grade']}")
        print(f"    Gaps - Accuracy: {d['utility_gaps']['accuracy_gap']}, F1: {d['utility_gaps']['f1_gap']}, AUC: {d['utility_gaps']['auc_gap']}")
        r = d["results"]
        print(f"    TRTR RF: acc={r['trtr_rf']['accuracy']}, TSTR RF: acc={r['tstr_rf']['accuracy']}")

# 7. Attack simulation
print("\n--- Attack Simulation ---")
if ds_id and job_id:
    atk_res = test("Privacy attack simulation", lambda: (
        requests.post(f"{BASE}/attacks/simulate", json={
            "dataset_id": ds_id, "synthetic_job_id": job_id
        }).json()
    ))
    if atk_res and atk_res.get("data"):
        d = atk_res["data"]
        print(f"    Overall Risk: {d['overall_risk_score']}/100 ({d['overall_risk_level']})")
        a = d["attacks"]
        print(f"    MIA: AUC={a['membership_inference']['attack_auc']}, risk={a['membership_inference']['risk_level']}")
        print(f"    Re-ID: {a['reidentification']['records_at_risk_pct']}% at risk, risk={a['reidentification']['risk_level']}")
        print(f"    Attr: advantage={a['attribute_inference']['average_advantage']}, risk={a['attribute_inference']['risk_level']}")

# 8. Federated learning
print("\n--- Federated Learning ---")
sample_csv = r"C:\Users\chira\.gemini\antigravity\scratch\synth-health-guard\data\sample\healthcare_data.csv"

fed_res = test("Create federation", lambda: (
    requests.post(f"{BASE}/federated/create", json={"total_rounds": 3}).json()
))
fed_id = fed_res["data"]["federation_id"] if fed_res and fed_res.get("data") else None

if fed_id:
    for name in ["City General", "Metro Health"]:
        test(f"Add hospital: {name}", lambda: (
            requests.post(f"{BASE}/federated/add-hospital",
                data={"federation_id": fed_id, "hospital_name": name},
                files={"file": ("data.csv", open(sample_csv, "rb"), "text/csv")}
            ).json()
        ))

    train_res = test("Federated training", lambda: (
        requests.post(f"{BASE}/federated/train", json={
            "federation_id": fed_id, "dp_epsilon": 1.0, "apply_dp": True
        }).json()
    ))
    if train_res and train_res.get("data"):
        d = train_res["data"]
        print(f"    Rounds: {d['rounds_completed']}, Order-independent: {d['order_independent_verified']}")

    fed_gen = test("Federated generate", lambda: (
        requests.post(f"{BASE}/federated/generate", json={
            "federation_id": fed_id, "num_rows": 200
        }).json()
    ))
    if fed_gen and fed_gen.get("data"):
        print(f"    Generated {fed_gen['data']['num_rows']} rows from {fed_gen['data']['num_hospitals']} hospitals")

# Summary
print(f"\n{'='*50}")
total = PASS_COUNT + FAIL_COUNT
print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {total} tests")
if FAIL_COUNT == 0:
    print("ALL TESTS PASSED!")
print(f"{'='*50}\n")
sys.exit(1 if FAIL_COUNT > 0 else 0)
