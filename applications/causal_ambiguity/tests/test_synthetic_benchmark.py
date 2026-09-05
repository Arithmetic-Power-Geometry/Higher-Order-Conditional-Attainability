from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import synthetic_benchmark as sb

def test_case_generation_is_deterministic():
    _, u1, b1, c1, a1, g1, cost1 = sb.generate_case(5, 'linear_gaussian', 250, 0)
    _, u2, b2, c2, a2, g2, cost2 = sb.generate_case(5, 'linear_gaussian', 250, 0)
    assert np.allclose(u1, u2)
    assert b1 == b2
    assert len(c1) == len(c2)
    assert np.allclose(a1, a2) and np.allclose(g1, g2) and np.allclose(cost1, cost2)

def test_oica_policy_produces_valid_trajectory():
    rows, stop = sb.run_policy('OICA', 5, 'linear_gaussian', 500, 2)
    assert len(rows) == sb.MAX_STEPS + 1
    assert 0 <= stop <= sb.MAX_STEPS + 1
    assert all(r['class_size'] >= 1 for r in rows)
    assert all(r['omega_task'] >= 0 for r in rows)

def test_benchmark_scope_constants():
    assert sb.EPS == 0.30
    assert sb.MAX_STEPS == 10
    assert set(sb.POLICIES) == {'OICA','ClassSize','Graph','Random'}
