import sys
import math
import numpy as np

sys.path.insert(0, "src")
from evcs.ue_evaluator import UEEvaluator, erlang_c, waiting_time_mms


# helpers

def make_tiny(N=12, seed=7):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(N, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    tau = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(tau, 0.0)
    d = rng.lognormal(mean=0.0, sigma=0.5, size=N)
    d = d / d.mean()
    return tau, d

def u_dict(servers):
    return {(j, 0): s for j, s in enumerate(servers)}

def ev(N=12):
    tau, d = make_tiny(N)
    return UEEvaluator(N=N, d=d, tau=tau, mu=1.2, s_max=4, noopt_cost=5.0), d


# queue tests

def test_erlang_c_saturated():
    assert erlang_c(1, 1.0, 1.0) == 1.0
    assert erlang_c(2, 4.0, 1.0) == 1.0

def test_erlang_c_light():
    assert erlang_c(4, 0.001, 1.0) < 0.01

def test_wt_saturated():
    assert waiting_time_mms(1, 1.0, 1.0) == 1.0e12

def test_wt_stable():
    wt = waiting_time_mms(2, 1.0, 0.5)
    assert wt > 0.0 and math.isfinite(wt)


# evaluator tests

def test_empty():
    e, _ = ev()
    score, _ = e.evaluate({}, T=1)
    assert score == 0.0

def test_finite():
    e, _ = ev()
    s = [0] * 12
    for j in [0, 3, 6, 9]:
        s[j] = 2
    score, _ = e.evaluate(u_dict(s), T=1)
    assert score > 0.0 and math.isfinite(score)
    return score

def test_monotone():
    e, _ = ev()
    s1 = [0] * 12
    s3 = [0] * 12
    for j in [0, 3, 6, 9]:
        s1[j] = 1
        s3[j] = 3
    sc1, _ = e.evaluate(u_dict(s1), T=1)
    sc3, _ = e.evaluate(u_dict(s3), T=1)
    assert sc3 >= sc1 - 1e-6
    return sc1, sc3

def test_all_open():
    e, d = ev()
    partial = [0] * 12
    for j in range(0, 12, 3):
        partial[j] = 2
    sp, _ = e.evaluate(u_dict(partial), T=1)
    sa, _ = e.evaluate(u_dict([4] * 12), T=1)
    assert sa >= sp - 1e-6
    assert sa <= float(np.sum(d)) + 1e-6
    return sp, sa

def test_cumulative():
    e, _ = ev()
    U2 = {}
    for j in [0, 3, 6, 9]:
        U2[(j, 0)] = 1
        U2[(j, 1)] = 1
    s2, _ = e.evaluate(U2, T=2, cumulative_install=True)
    sd, _ = e.evaluate({(j, 0): 2 for j in [0, 3, 6, 9]}, T=1)
    assert abs(s2 - sd) < 1e-4
    return s2, sd

def test_ctx():
    tau, d = make_tiny()
    with UEEvaluator(N=12, d=d, tau=tau, mu=1.2, s_max=4, noopt_cost=5.0) as e:
        score, _ = e.evaluate({(j, 0): 2 for j in range(12)}, T=1)
    assert score > 0
    return score


# run

if __name__ == "__main__":
    test_erlang_c_saturated();  print("erlang_c saturated  OK")
    test_erlang_c_light();      print("erlang_c light      OK")
    test_wt_saturated();        print("wt saturated        OK")
    test_wt_stable();           print("wt stable           OK")

    test_empty();               print("empty               OK")
    sc = test_finite();         print(f"finite              OK  score={sc:.4f}")
    sc1, sc3 = test_monotone(); print(f"monotone            OK  1s={sc1:.4f} 3s={sc3:.4f}")
    sp, sa = test_all_open();   print(f"all-open            OK  partial={sp:.4f} all={sa:.4f}")
    s2, sd = test_cumulative(); print(f"cumulative          OK  T2={s2:.4f} direct={sd:.4f}")
    sc = test_ctx();            print(f"context manager     OK  score={sc:.4f}")

    print("\nAll tests passed.")
