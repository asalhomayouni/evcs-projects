from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Single shared environment — created once per process so that every
# gp.Model() call reuses the same license slot instead of checking out
# a new one each time (important on Compute Canada cluster licenses).
_grb_env: gp.Env | None = None


def _get_env() -> gp.Env:
    global _grb_env
    if _grb_env is None:
        _grb_env = gp.Env()
    return _grb_env


# --- key normaliser ---
def _nk(k):
    return tuple(int(x) for x in k) if isinstance(k, tuple) else int(k)


# --- proxies ---
class _Scalar:
    __slots__ = ("_v",)
    def __init__(self, v): self._v = float(v)
    @property
    def value(self): return self._v
    def __float__(self): return self._v
    def __int__(self): return int(self._v)
    def __index__(self): return int(self._v)
    def __call__(self): return self._v


class _IndexProxy:
    __slots__ = ("_d",)
    def __init__(self, data): self._d = {_nk(k): float(v) for k, v in data.items()}
    def __getitem__(self, k): return self._d[_nk(k)]
    def __call__(self, *a): return self[a[0] if len(a) == 1 else a]


class _VarProxy:
    __slots__ = ("_var", "_manual", "_use_manual")
    def __init__(self, v):
        object.__setattr__(self, "_var", v)
        object.__setattr__(self, "_manual", None)
        object.__setattr__(self, "_use_manual", False)

    @property
    def value(self):
        if not self._use_manual:
            try:
                x = self._var.X
                if x is not None: return x
            except (gp.GurobiError, AttributeError): pass
        return self._manual

    @value.setter
    def value(self, v):
        object.__setattr__(self, "_manual", v)
        object.__setattr__(self, "_use_manual", v is not None)
        if v is not None:
            try: self._var.Start = float(v)
            except Exception: pass

    def __float__(self):
        v = self.value; return float(v) if v is not None else 0.0
    def __call__(self):
        v = self.value; return v if v is not None else 0.0


class _VarDict:
    def __init__(self, d): self._p = {_nk(k): _VarProxy(v) for k, v in d.items()}
    def __getitem__(self, k): return self._p[_nk(k)]
    def __setitem__(self, k, v): self._p[_nk(k)].value = v
    def __iter__(self): return iter(self._p)
    def __contains__(self, k): return _nk(k) in self._p
    def __bool__(self): return bool(self._p)
    def __len__(self): return len(self._p)
    def _reset(self):
        for p in self._p.values(): object.__setattr__(p, "_use_manual", False)


# --- wrapper ---
class GRBModelWrapper:
    def __init__(self, gm, meta):
        self._gm, self._meta = gm, meta
        self.I = range(meta.M)
        self.J = range(meta.N)
        self.Arcs = meta.arcs
        if meta.T > 0:
            self.T = range(meta.T)
        self.Q = _Scalar(meta.Q)
        self.U = _Scalar(meta.site_ub)
        self.a = _IndexProxy(meta.a)
        self.P = (_IndexProxy({t: float(meta.P_T[t]) for t in range(meta.T)})
                  if meta.T > 0 else _Scalar(float(meta.P_single)))
        self.x = _VarDict(meta.x)
        self.y = _VarDict(meta.y)
        if meta.z: self.z = _VarDict(meta.z)
        if meta.u: self.u = _VarDict(meta.u)

    def _mark_solved(self):
        for attr in ("x", "y", "z", "u"):
            vd = getattr(self, attr, None)
            if isinstance(vd, _VarDict): vd._reset()

    def clone(self):
        m = self._meta
        if m.T > 0:
            return build_multi_period_model_grb(
                M=m.M, N=m.N, T=m.T, in_range=m.arcs, Ji=m.Ji, Ij=m.Ij,
                demand_IT=m.demand_arr, Q=m.Q, P_T=m.P_T, distIJ=m.distIJ,
                method_name=m.method_name, allow_multi_charger=m.allow_multi_charger,
                max_chargers_per_site=m.max_chargers_per_site,
                cumulative_install=m.cumulative_install,
            )
        return build_base_model_grb(
            M=m.M, N=m.N, in_range=m.arcs, Ji=m.Ji, Ij=m.Ij,
            demand_I=m.demand_arr, Q=m.Q, P=m.P_single, distIJ=m.distIJ,
            method_name=m.method_name, allow_multi_charger=m.allow_multi_charger,
            max_chargers_per_site=m.max_chargers_per_site,
        )

    class _S:
        def load_from(self, *a, **kw): pass
    solutions = _S()

    def __repr__(self):
        return f"<GRBModelWrapper M={self._meta.M} N={self._meta.N} T={self._meta.T}>"


# --- helpers ---
def _site_ub(budget, mc): return max(0, int(budget) if mc is None else int(mc))

def _farther(distIJ, Ji):
    fo = {}
    for i, js in Ji.items():
        s = sorted(js, key=lambda j: distIJ[i][j])
        for pos, j in enumerate(s):
            fo[(i, j)] = [(i, k) for k in s[pos+1:]]
    return fo

def _method_single(gm, name, distIJ, arcs, Ji, opv, y, a):
    n = str(name).lower()
    if n == "closest_only":
        fo = _farther(distIJ, Ji)
        for (i, j), far in fo.items():
            if far:
                gm.addConstr(gp.quicksum(y[ii,jj] for ii,jj in far) <= 1 - opv[j])
    elif n == "closest_priority":
        gm.setObjective(
            gp.quicksum(a[i]*y[i,j] for i,j in arcs) +
            1e-3*gp.quicksum((1-float(distIJ[i][j]))*y[i,j] for i,j in arcs),
            GRB.MAXIMIZE)
    elif n == "system_optimum":
        gm.setObjective(
            gp.quicksum(a[i]*y[i,j] for i,j in arcs) -
            0.1*gp.quicksum(float(distIJ[i][j])*y[i,j] for i,j in arcs),
            GRB.MAXIMIZE)

def _method_multi(gm, name, distIJ, arcs, Ji, z, y, a, T):
    n = str(name).lower()
    Tl = list(T)
    if n == "closest_only":
        fo = _farther(distIJ, Ji)
        for t in Tl:
            for (i, j), far in fo.items():
                if far:
                    gm.addConstr(gp.quicksum(y[ii,jj,t] for ii,jj in far) <= 1 - z[j,t])
    elif n == "closest_priority":
        gm.setObjective(
            gp.quicksum(a[i,t]*y[i,j,t] for i,j in arcs for t in Tl) +
            1e-3*gp.quicksum((1-float(distIJ[i][j]))*y[i,j,t] for i,j in arcs for t in Tl),
            GRB.MAXIMIZE)
    elif n == "system_optimum":
        gm.setObjective(
            gp.quicksum(a[i,t]*y[i,j,t] for i,j in arcs for t in Tl) -
            0.1*gp.quicksum(float(distIJ[i][j])*y[i,j,t] for i,j in arcs for t in Tl),
            GRB.MAXIMIZE)


# --- single-period builder ---
def build_base_model_grb(M, N, in_range, Ji, Ij, demand_I, Q, P,
                          distIJ=None, method_name="none",
                          allow_multi_charger=False, max_chargers_per_site=None):
    gm = gp.Model(env=_get_env()); gm.setParam("OutputFlag", 0)
    arcs = list(in_range)
    ub = _site_ub(P, max_chargers_per_site)
    a = {i: float(demand_I[i]) for i in range(M)}

    x, z = {}, {}
    if allow_multi_charger:
        for j in range(N):
            z[j] = gm.addVar(vtype=GRB.BINARY, name=f"z_{j}")
            x[j] = gm.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"x_{j}")
    else:
        for j in range(N):
            x[j] = gm.addVar(vtype=GRB.BINARY, name=f"x_{j}")
    y = {(i,j): gm.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for i,j in arcs}
    gm.update()

    gm.setObjective(gp.quicksum(a[i]*y[i,j] for i,j in arcs), GRB.MAXIMIZE)

    for i in range(M):
        if Ji.get(i): gm.addConstr(gp.quicksum(y[i,j] for j in Ji[i]) <= 1)
    if allow_multi_charger:
        for i,j in arcs: gm.addConstr(y[i,j] <= z[j])
        for j in range(N): gm.addConstr(x[j] <= ub*z[j])
        for j in range(N):
            if Ij.get(j): gm.addConstr(gp.quicksum(a[i]*y[i,j] for i in Ij[j]) <= Q*x[j])
        gm.addConstr(gp.quicksum(x[j] for j in range(N)) <= P)
    else:
        for i,j in arcs: gm.addConstr(y[i,j] <= x[j])
        for j in range(N):
            if Ij.get(j): gm.addConstr(gp.quicksum(a[i]*y[i,j] for i in Ij[j]) <= Q*x[j])
        gm.addConstr(gp.quicksum(x[j] for j in range(N)) <= P)

    if str(method_name).lower() not in ("none","base") and distIJ is not None:
        _method_single(gm, method_name, distIJ, arcs, Ji, z if allow_multi_charger else x, y, a)

    meta = SimpleNamespace(
        M=M, N=N, T=0, arcs=arcs, Ji=dict(Ji), Ij=dict(Ij),
        Q=float(Q), P_single=int(P), P_T=None, site_ub=ub,
        allow_multi_charger=allow_multi_charger, cumulative_install=False,
        method_name=str(method_name), max_chargers_per_site=max_chargers_per_site,
        distIJ=distIJ, demand_arr=np.asarray(demand_I, dtype=float),
        x=x, y=y, z=z, u={}, a=a,
    )
    return GRBModelWrapper(gm, meta)


# --- multi-period builder ---
def build_multi_period_model_grb(M, N, T, in_range, Ji, Ij, demand_IT, Q, P_T,
                                  distIJ=None, method_name="none",
                                  allow_multi_charger=True, max_chargers_per_site=None,
                                  cumulative_install=True):
    P_T = list(P_T)
    if len(P_T) != T: raise ValueError(f"P_T length must be {T}")
    gm = gp.Model(env=_get_env()); gm.setParam("OutputFlag", 0)
    arcs = list(in_range)
    ub = _site_ub(max(P_T), max_chargers_per_site)
    d = np.asarray(demand_IT, dtype=float)
    a = {(i,t): float(d[t,i]) for i in range(M) for t in range(T)}

    x, y, z, u = {}, {}, {}, {}
    if not allow_multi_charger:
        for j in range(N):
            for t in range(T): x[j,t] = gm.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")
        for i,j in arcs:
            for t in range(T): y[i,j,t] = gm.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{t}")
    else:
        for j in range(N):
            for t in range(T):
                u[j,t] = gm.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"u_{j}_{t}")
                x[j,t] = gm.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"x_{j}_{t}")
                z[j,t] = gm.addVar(vtype=GRB.BINARY, name=f"z_{j}_{t}")
        for i,j in arcs:
            for t in range(T): y[i,j,t] = gm.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{t}")
    gm.update()

    gm.setObjective(gp.quicksum(a[i,t]*y[i,j,t] for i,j in arcs for t in range(T)), GRB.MAXIMIZE)

    for i in range(M):
        if Ji.get(i):
            for t in range(T): gm.addConstr(gp.quicksum(y[i,j,t] for j in Ji[i]) <= 1)

    if not allow_multi_charger:
        for i,j in arcs:
            for t in range(T): gm.addConstr(y[i,j,t] <= x[j,t])
        for j in range(N):
            for t in range(T):
                if Ij.get(j): gm.addConstr(gp.quicksum(a[i,t]*y[i,j,t] for i in Ij[j]) <= Q*x[j,t])
        for t in range(T): gm.addConstr(gp.quicksum(x[j,t] for j in range(N)) <= P_T[t])
    else:
        for j in range(N):
            for t in range(T):
                gm.addConstr(x[j,t] >= z[j,t])
                gm.addConstr(x[j,t] <= ub*z[j,t])
        for i,j in arcs:
            for t in range(T): gm.addConstr(y[i,j,t] <= z[j,t])
        for j in range(N):
            for t in range(T):
                if Ij.get(j): gm.addConstr(gp.quicksum(a[i,t]*y[i,j,t] for i in Ij[j]) <= Q*x[j,t])
        for j in range(N):
            for t in range(T):
                if cumulative_install:
                    gm.addConstr(x[j,t] == (u[j,t] if t==0 else x[j,t-1]+u[j,t]))
                else:
                    gm.addConstr(x[j,t] == u[j,t])
        for t in range(T): gm.addConstr(gp.quicksum(u[j,t] for j in range(N)) <= P_T[t])

    if str(method_name).lower() not in ("none","base") and distIJ is not None:
        _method_multi(gm, method_name, distIJ, arcs, Ji, z, y, a, range(T))

    meta = SimpleNamespace(
        M=M, N=N, T=T, arcs=arcs, Ji=dict(Ji), Ij=dict(Ij),
        Q=float(Q), P_single=None, P_T=list(P_T), site_ub=ub,
        allow_multi_charger=allow_multi_charger, cumulative_install=cumulative_install,
        method_name=str(method_name), max_chargers_per_site=max_chargers_per_site,
        distIJ=distIJ, demand_arr=d,
        x=x, y=y, z=z, u=u, a=a,
    )
    return GRBModelWrapper(gm, meta)
