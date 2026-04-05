# Tests

## Running Tests

```bash
# Phase 0: sanity check (imports + synthetic data)
python tests/test_pipeline.py

# Phase 1: full module validation (41 checks)
python tests/test_phase1_validation.py
```

## Coverage

| Test File | Modules Covered | Checks |
|-----------|----------------|--------|
| `test_pipeline.py` | All 8 core modules (synthetic data) | imports, gen_uncertainty, gp_oracle, fusion, calibration, ood, evaluate |
| `test_phase1_validation.py` | Per-module functionality, data I/O, numerical stability, API contracts | 41 checks |
