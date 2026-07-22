/* MLKit — faithful, deterministic in-browser ML for the playground.
   All models operate on data normalized to [0,1]^2 (or 1-D). Fitting standardizes
   internally where needed and returns everything the UI must draw, so the UI never
   touches transforms. Datasets are cached per kind so moving a slider changes only
   the hyper-parameter, never the data. */
(function () {
  "use strict";

  // ---- deterministic RNG (mulberry32) ----
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const clip01 = x => Math.max(0.02, Math.min(0.98, x));

  // ---- datasets (cached) ----
  const _cache = {};
  function dataset(kind) {
    if (_cache[kind]) return _cache[kind];
    let out;
    if (kind === "reg1d") {
      const r = rng(101), X = [], y = [];
      for (let i = 0; i < 42; i++) {
        const x = 0.05 + 0.9 * (i / 41);
        const truth = 0.5 + 0.34 * Math.sin(x * 6 + 0.3) - 0.12 * x;
        X.push([x]); y.push(clip01(truth + gauss(r) * 0.055));
      }
      out = { X, y };
    } else if (kind === "blobs2") {
      const r = rng(202), X = [], y = [];
      const C = [[0.34, 0.44], [0.66, 0.58]];
      for (let c = 0; c < 2; c++) for (let i = 0; i < 55; i++) {
        X.push([clip01(C[c][0] + gauss(r) * 0.1), clip01(C[c][1] + gauss(r) * 0.1)]); y.push(c);
      }
      out = { X, y };
    } else if (kind === "gnbhet") { // heterocedástico: clase compacta vs difusa (visibiliza var_smoothing en GaussianNB)
      const r = rng(202), X = [], y = [];
      for (let i = 0; i < 55; i++) { X.push([clip01(0.5 + gauss(r) * 0.06), clip01(0.5 + gauss(r) * 0.06)]); y.push(0); }
      for (let i = 0; i < 55; i++) { X.push([clip01(0.5 + gauss(r) * 0.19), clip01(0.5 + gauss(r) * 0.19)]); y.push(1); }
      out = { X, y };
    } else if (kind === "moons") {
      const r = rng(303), X = [], y = [];
      for (let i = 0; i < 60; i++) {
        const t = Math.PI * (i / 59);
        X.push([clip01(0.32 + 0.30 * Math.cos(t) + gauss(r) * 0.03), clip01(0.60 - 0.30 * Math.sin(t) + gauss(r) * 0.03)]); y.push(0);
        X.push([clip01(0.68 - 0.30 * Math.cos(t) + gauss(r) * 0.03), clip01(0.40 + 0.30 * Math.sin(t) + gauss(r) * 0.03)]); y.push(1);
      }
      out = { X, y };
    } else if (kind === "blobs3") {
      const r = rng(404), X = [];
      const C = [[0.28, 0.34], [0.72, 0.32], [0.5, 0.72]];
      for (let c = 0; c < 3; c++) for (let i = 0; i < 34; i++)
        X.push([clip01(C[c][0] + gauss(r) * 0.075), clip01(C[c][1] + gauss(r) * 0.075)]);
      out = { X };
    } else if (kind === "blobs3n") { // 3 blobs + uniform noise (dbscan)
      const base = dataset("blobs3");
      const r = rng(505), X = base.X.map(p => p.slice());
      for (let i = 0; i < 10; i++) X.push([0.06 + r() * 0.88, 0.06 + r() * 0.88]);
      out = { X };
    } else if (kind === "cloud") { // elongated cloud for PCA
      const r = rng(606), X = [], ang = 32 * Math.PI / 180, ca = Math.cos(ang), sa = Math.sin(ang);
      for (let i = 0; i < 70; i++) {
        const u = gauss(r) * 0.26, v = gauss(r) * 0.075;
        X.push([clip01(0.5 + u * ca - v * sa), clip01(0.5 + u * sa + v * ca)]);
      }
      out = { X };
    } else if (kind === "moonsn") { // two-moons ruidoso + flips de etiqueta (boosting: overfit visible)
      const r = rng(717), X = [], y = [];
      for (let i = 0; i < 55; i++) {
        const t = Math.PI * (i / 54);
        X.push([clip01(0.32 + 0.30 * Math.cos(t) + gauss(r) * 0.075), clip01(0.58 - 0.30 * Math.sin(t) + gauss(r) * 0.075)]); y.push(0);
        X.push([clip01(0.68 - 0.30 * Math.cos(t) + gauss(r) * 0.075), clip01(0.42 + 0.30 * Math.sin(t) + gauss(r) * 0.075)]); y.push(1);
      }
      for (let k = 0; k < Math.round(X.length * 0.08); k++) { const j = Math.floor(r() * X.length); y[j] = 1 - y[j]; }
      out = { X, y };
    } else if (kind === "vardens") { // densidad variable: núcleo denso + 2 lóbulos dispersos + ruido (HDBSCAN)
      const r = rng(717), X = [];
      for (let i = 0; i < 30; i++) X.push([clip01(0.29 + gauss(r) * 0.045), clip01(0.40 + gauss(r) * 0.045)]);
      for (let i = 0; i < 17; i++) X.push([clip01(0.63 + gauss(r) * 0.055), clip01(0.60 + gauss(r) * 0.06)]);
      for (let i = 0; i < 17; i++) X.push([clip01(0.80 + gauss(r) * 0.055), clip01(0.66 + gauss(r) * 0.06)]);
      for (let i = 0; i < 12; i++) X.push([0.05 + r() * 0.9, 0.05 + r() * 0.9]);
      out = { X };
    } else if (kind === "arc") { // variedad 1-D curvada (arco) inmersa en 2-D (autoencoder vs PCA)
      const r = rng(717), X = [];
      for (let i = 0; i < 46; i++) {
        const t = i / 45, ang = Math.PI * (0.08 + 0.84 * t);
        X.push([clip01(0.5 + 0.34 * Math.cos(ang) + gauss(r) * 0.022), clip01(0.24 + 0.44 * Math.sin(ang) + gauss(r) * 0.022)]);
      }
      out = { X };
    } else if (kind === "prophetTs") { // serie 1-D: tendencia en V invertida + estacionalidad creciente (Prophet)
      const r = rng(707), X = [], y = [];
      const N = 64, cp = 0.68, f = 4;
      const level = t => t <= cp ? 0.35 + 0.45 * t : 0.35 + 0.45 * cp - 0.95 * (t - cp);
      for (let i = 0; i < N; i++) {
        const t = i / (N - 1), L = level(t), amp = 0.16 * L;
        X.push([t]); y.push(clip01(L + amp * Math.sin(2 * Math.PI * f * t) + gauss(r) * 0.02));
      }
      out = { X, y };
    }
    _cache[kind] = out;
    return out;
  }

  // ---- helpers ----
  function standardize(X) {
    const d = X[0].length, mean = Array(d).fill(0), std = Array(d).fill(0);
    X.forEach(row => row.forEach((v, j) => mean[j] += v));
    mean.forEach((_, j) => mean[j] /= X.length);
    X.forEach(row => row.forEach((v, j) => std[j] += (v - mean[j]) ** 2));
    std.forEach((_, j) => std[j] = Math.sqrt(std[j] / X.length) || 1);
    const Z = X.map(row => row.map((v, j) => (v - mean[j]) / std[j]));
    return { Z, mean, std, apply: p => p.map((v, j) => (v - mean[j]) / std[j]) };
  }
  const GRID = 46;
  function classGrid(predict) {
    const g = [];
    for (let j = 0; j < GRID; j++) { const row = []; for (let i = 0; i < GRID; i++) row.push(predict([(i + 0.5) / GRID, (j + 0.5) / GRID])); g.push(row); }
    return g;
  }
  function accuracy(X, y, predict) { let ok = 0; for (let i = 0; i < X.length; i++) if (predict(X[i]) === y[i]) ok++; return ok / X.length; }

  // ============ SUPERVISED ============

  // Ridge / polynomial regression (closed form, intercept unpenalized)
  function ridgePoly(alpha, degree, fitIntercept) {
    const { X, y } = dataset("reg1d");
    const feat = x => { const f = fitIntercept ? [1] : []; for (let d = 1; d <= degree; d++) f.push(Math.pow(x, d)); return f; };
    const P = X.map(row => feat(row[0])), n = P.length, m = P[0].length;
    // A = PᵀP + αI' (skip intercept col), b = Pᵀy
    const A = Array.from({ length: m }, () => Array(m).fill(0)), b = Array(m).fill(0);
    for (let i = 0; i < n; i++) for (let a = 0; a < m; a++) { b[a] += P[i][a] * y[i]; for (let c = 0; c < m; c++) A[a][c] += P[i][a] * P[i][c]; }
    for (let a = 0; a < m; a++) { const isIntercept = fitIntercept && a === 0; if (!isIntercept) A[a][a] += alpha; }
    const w = solve(A, b);
    const predict = x => { const f = feat(x); let s = 0; for (let a = 0; a < m; a++) s += w[a] * f[a]; return s; };
    const ybar = y.reduce((s, v) => s + v, 0) / n;
    let ssRes = 0, ssTot = 0; for (let i = 0; i < n; i++) { ssRes += (y[i] - predict(X[i][0])) ** 2; ssTot += (y[i] - ybar) ** 2; }
    const curve = []; for (let i = 0; i <= 80; i++) { const x = i / 80; curve.push({ x, y: predict(x) }); }
    return { kind: "curve", points: X.map((row, i) => ({ x: row[0], y: y[i] })), curve, metric: { key: "R²", val: (ssTot ? 1 - ssRes / ssTot : 0).toFixed(3) } };
  }
  function solve(A, b) { // Gaussian elimination
    const n = b.length, M = A.map((row, i) => row.concat(b[i]));
    for (let c = 0; c < n; c++) {
      let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c] || 1e-9;
      for (let k = c; k <= n; k++) M[c][k] /= d;
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map(row => row[n]);
  }

  // Logistic regression (L2, sklearn-style: min 0.5||w||^2 + C·Σ logloss)
  function logreg(C, penalty) {
    const { X, y } = dataset("blobs2"), S = standardize(X), Z = S.Z, n = Z.length;
    let w = [0, 0], b = 0; const lr = 0.3, iters = 500, lam = penalty === "none" ? 0 : 1 / C;
    const sig = z => 1 / (1 + Math.exp(-z));
    for (let it = 0; it < iters; it++) {
      let gw = [lam * w[0], lam * w[1]], gb = 0;
      for (let i = 0; i < n; i++) { const p = sig(w[0] * Z[i][0] + w[1] * Z[i][1] + b), e = p - y[i]; gw[0] += e * Z[i][0]; gw[1] += e * Z[i][1]; gb += e; }
      w[0] -= lr * gw[0] / n; w[1] -= lr * gw[1] / n; b -= lr * gb / n;
    }
    const predict = p => { const z = S.apply(p); return sig(w[0] * z[0] + w[1] * z[1] + b) > 0.5 ? 1 : 0; };
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  // Gaussian Naive Bayes (exact)
  function gnb(varSmoothing) {
    const { X, y } = dataset("gnbhet"), n = X.length, d = 2, classes = [0, 1];
    const dim = [0, 1];
    const stats = classes.map(c => {
      const rows = X.filter((_, i) => y[i] === c), mean = [0, 0], varr = [0, 0];
      rows.forEach(r => dim.forEach(j => mean[j] += r[j])); dim.forEach(j => mean[j] /= rows.length);
      rows.forEach(r => dim.forEach(j => varr[j] += (r[j] - mean[j]) ** 2)); dim.forEach(j => varr[j] /= rows.length);
      return { c, mean, varr, prior: rows.length / n, rows: rows.length };
    });
    // eps = var_smoothing · max_j Var_global(feature j)  (como sklearn: X.var(axis=0).max())
    let maxVar = 0;
    for (let j = 0; j < d; j++) { const m = X.reduce((s, rr) => s + rr[j], 0) / n; const vv = X.reduce((s, rr) => s + (rr[j] - m) ** 2, 0) / n; if (vv > maxVar) maxVar = vv; }
    const eps = varSmoothing * maxVar;
    const predict = p => {
      let best = -Infinity, bc = 0;
      stats.forEach(s => {
        let lp = Math.log(s.prior);
        for (let j = 0; j < d; j++) { const v = s.varr[j] + eps; lp += -0.5 * Math.log(2 * Math.PI * v) - (p[j] - s.mean[j]) ** 2 / (2 * v); }
        if (lp > best) { best = lp; bc = s.c; }
      });
      return bc;
    };
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  // Linear SVM, soft margin (primal subgradient): 0.5||w||^2 + C·Σ hinge
  function svmLinear(C) {
    const { X, y } = dataset("blobs2"), S = standardize(X), Z = S.Z, n = Z.length;
    const t = y.map(v => v === 1 ? 1 : -1);
    let w = [0, 0], b = 0; const lr = 0.02, iters = 400;
    for (let it = 0; it < iters; it++) {
      let gw = [w[0], w[1]], gb = 0;
      for (let i = 0; i < n; i++) { const m = t[i] * (w[0] * Z[i][0] + w[1] * Z[i][1] + b); if (m < 1) { gw[0] -= C * t[i] * Z[i][0]; gw[1] -= C * t[i] * Z[i][1]; gb -= C * t[i]; } }
      w[0] -= lr * gw[0] / n; w[1] -= lr * gw[1] / n; b -= lr * gb / n;
    }
    const f = p => { const z = S.apply(p); return w[0] * z[0] + w[1] * z[1] + b; };
    const predict = p => f(p) > 0 ? 1 : 0;
    // boundary + margins in data space: sample the three lines f=0,±1 across x
    const lines = { mid: [], up: [], dn: [] };
    for (let i = 0; i <= 60; i++) {
      const x = i / 60, zx = (x - S.mean[0]) / S.std[0];
      const zy = level => (level - b - w[0] * zx) / (w[1] || 1e-9);
      lines.mid.push({ x, y: zy(0) * S.std[1] + S.mean[1] });
      lines.up.push({ x, y: zy(1) * S.std[1] + S.mean[1] });
      lines.dn.push({ x, y: zy(-1) * S.std[1] + S.mean[1] });
    }
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), lines, metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  // ---- generic CART (classification: gini/entropy; regression: variance) ----
  function buildTree(X, y, opt) {
    const task = opt.task, maxDepth = opt.maxDepth, minLeaf = opt.minLeaf || 1, maxFeatures = opt.maxFeatures || X[0].length, rand = opt.rand;
    const nFeat = X[0].length;
    function impurity(idx) {
      if (task === "reg") { const m = idx.reduce((s, i) => s + y[i], 0) / idx.length; return idx.reduce((s, i) => s + (y[i] - m) ** 2, 0) / idx.length; }
      const cnt = {}; idx.forEach(i => cnt[y[i]] = (cnt[y[i]] || 0) + 1); let imp = 0;
      Object.values(cnt).forEach(c => { const p = c / idx.length; imp += opt.criterion === "entropy" ? -p * Math.log2(p) : p * (1 - p); });
      return imp;
    }
    function leaf(idx) {
      if (task === "reg") return { leaf: true, val: idx.reduce((s, i) => s + y[i], 0) / idx.length };
      const cnt = {}; idx.forEach(i => cnt[y[i]] = (cnt[y[i]] || 0) + 1); let bc = null, bn = -1;
      Object.keys(cnt).forEach(k => { if (cnt[k] > bn) { bn = cnt[k]; bc = +k; } }); return { leaf: true, val: bc };
    }
    function feats() {
      if (maxFeatures >= nFeat) return Array.from({ length: nFeat }, (_, i) => i);
      const pool = Array.from({ length: nFeat }, (_, i) => i), pick = [];
      while (pick.length < maxFeatures && pool.length) pick.push(pool.splice(Math.floor((rand ? rand() : Math.random()) * pool.length), 1)[0]);
      return pick;
    }
    function grow(idx, depth) {
      if (idx.length < 2 * minLeaf || depth >= maxDepth || impurity(idx) < 1e-9) return leaf(idx);
      const parent = impurity(idx); let best = null;
      feats().forEach(f => {
        const vals = idx.map(i => X[i][f]).sort((a, b) => a - b), cands = [];
        for (let k = 0; k < vals.length - 1; k++) if (vals[k] !== vals[k + 1]) cands.push((vals[k] + vals[k + 1]) / 2);
        const step = Math.max(1, Math.floor(cands.length / 24));
        for (let k = 0; k < cands.length; k += step) {
          const thr = cands[k], L = [], R = [];
          idx.forEach(i => (X[i][f] <= thr ? L : R).push(i));
          if (L.length < minLeaf || R.length < minLeaf) continue;
          const gain = parent - (L.length * impurity(L) + R.length * impurity(R)) / idx.length;
          if (!best || gain > best.gain) best = { gain, f, thr, L, R };
        }
      });
      if (!best || best.gain <= 1e-9) return leaf(idx);
      return { leaf: false, f: best.f, thr: best.thr, left: grow(best.L, depth + 1), right: grow(best.R, depth + 1) };
    }
    const root = grow(X.map((_, i) => i), 0);
    return p => { let n = root; while (!n.leaf) n = p[n.f] <= n.thr ? n.left : n.right; return n.val; };
  }

  function tree(maxDepth, minLeaf, criterion) {
    const { X, y } = dataset("moons");
    const predict = buildTree(X, y, { task: "clf", maxDepth, minLeaf, criterion });
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  function forest(nEst, maxDepth, maxFeaturesMode) {
    const { X, y } = dataset("moons"), n = X.length, r = rng(9090);
    const maxFeatures = maxFeaturesMode === "sqrt" ? Math.max(1, Math.round(Math.sqrt(2))) : 2;
    const trees = [];
    for (let b = 0; b < nEst; b++) {
      const bx = [], by = []; for (let i = 0; i < n; i++) { const j = Math.floor(r() * n); bx.push(X[j]); by.push(y[j]); }
      trees.push(buildTree(bx, by, { task: "clf", maxDepth, minLeaf: 1, criterion: "gini", maxFeatures, rand: r }));
    }
    const predict = p => { let v = 0; trees.forEach(t => v += t(p)); return v * 2 >= nEst ? 1 : 0; };
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  function gboost(nEst, lr, maxDepth) {
    const { X, y } = dataset("reg1d"), n = X.length;
    const base = y.reduce((s, v) => s + v, 0) / n, trees = [];
    let F = X.map(() => base);
    for (let m = 0; m < nEst; m++) {
      const resid = y.map((v, i) => v - F[i]);
      const t = buildTree(X, resid, { task: "reg", maxDepth, minLeaf: 1 });
      trees.push(t); for (let i = 0; i < n; i++) F[i] += lr * t(X[i]);
    }
    const predict = x => { let v = base; trees.forEach(t => v += lr * t([x])); return v; };
    let mse = 0; for (let i = 0; i < n; i++) mse += (y[i] - F[i]) ** 2; mse /= n;
    const ybar = base; let ssTot = 0; for (let i = 0; i < n; i++) ssTot += (y[i] - ybar) ** 2;
    const curve = []; for (let i = 0; i <= 80; i++) { const x = i / 80; curve.push({ x, y: predict(x) }); }
    return { kind: "curve", points: X.map((row, i) => ({ x: row[0], y: y[i] })), curve, metric: { key: "train MSE", val: mse.toFixed(4) }, metric2: { key: "R²", val: (ssTot ? 1 - mse * n / ssTot : 0).toFixed(3) } };
  }

  // ---- MLP (1 hidden layer, sigmoid out, BCE, full-batch GD) ----
  function mlp(hidden, activation, lr, epochs) {
    const { X, y } = dataset("moons"), S = standardize(X), Z = S.Z, n = Z.length, r = rng(1234);
    const act = activation === "tanh" ? (Math.tanh) : (x => Math.max(0, x));
    const dact = activation === "tanh" ? (a => 1 - a * a) : (a => a > 0 ? 1 : 0);
    let W1 = Array.from({ length: hidden }, () => [gauss(r) * 0.6, gauss(r) * 0.6]);
    let b1 = Array(hidden).fill(0);
    let W2 = Array.from({ length: hidden }, () => gauss(r) * 0.6), b2 = 0;
    const sig = z => 1 / (1 + Math.exp(-z));
    for (let ep = 0; ep < epochs; ep++) {
      const gW1 = W1.map(() => [0, 0]), gb1 = Array(hidden).fill(0), gW2 = Array(hidden).fill(0); let gb2 = 0;
      for (let i = 0; i < n; i++) {
        const a1 = new Array(hidden); for (let h = 0; h < hidden; h++) a1[h] = act(W1[h][0] * Z[i][0] + W1[h][1] * Z[i][1] + b1[h]);
        let o = b2; for (let h = 0; h < hidden; h++) o += W2[h] * a1[h];
        const p = sig(o), e = p - y[i];
        for (let h = 0; h < hidden; h++) { gW2[h] += e * a1[h]; const dh = e * W2[h] * dact(a1[h]); gW1[h][0] += dh * Z[i][0]; gW1[h][1] += dh * Z[i][1]; gb1[h] += dh; }
        gb2 += e;
      }
      for (let h = 0; h < hidden; h++) { W2[h] -= lr * gW2[h] / n; b1[h] -= lr * gb1[h] / n; W1[h][0] -= lr * gW1[h][0] / n; W1[h][1] -= lr * gW1[h][1] / n; }
      b2 -= lr * gb2 / n;
    }
    const predict = p => {
      const z = S.apply(p); let o = b2;
      for (let h = 0; h < hidden; h++) o += W2[h] * act(W1[h][0] * z[0] + W1[h][1] * z[1] + b1[h]);
      return sig(o) > 0.5 ? 1 : 0;
    };
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  // ============ UNSUPERVISED ============
  function dist2(a, b) { return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2; }

  function kmeansppInit(X, k, r) {
    const centers = [X[Math.floor(r() * X.length)].slice()];
    while (centers.length < k) {
      const d = X.map(p => Math.min(...centers.map(c => dist2(p, c))));
      const sum = d.reduce((s, v) => s + v, 0) || 1; let t = r() * sum, i = 0;
      while (t > d[i] && i < d.length - 1) { t -= d[i]; i++; }
      centers.push(X[i].slice());
    }
    return centers;
  }
  function kmeans(k, nInit) {
    const { X } = dataset("blobs3"); let best = null;
    for (let run = 0; run < nInit; run++) {
      const r = rng(700 + run * 13); let C = kmeansppInit(X, k, r), labels = new Array(X.length).fill(0);
      for (let it = 0; it < 60; it++) {
        let moved = false;
        for (let i = 0; i < X.length; i++) { let bd = Infinity, bc = 0; for (let c = 0; c < k; c++) { const d = dist2(X[i], C[c]); if (d < bd) { bd = d; bc = c; } } if (labels[i] !== bc) { labels[i] = bc; moved = true; } }
        const sum = Array.from({ length: k }, () => [0, 0]), cnt = Array(k).fill(0);
        for (let i = 0; i < X.length; i++) { sum[labels[i]][0] += X[i][0]; sum[labels[i]][1] += X[i][1]; cnt[labels[i]]++; }
        for (let c = 0; c < k; c++) if (cnt[c]) C[c] = [sum[c][0] / cnt[c], sum[c][1] / cnt[c]];
        if (!moved && it > 0) break;
      }
      let inertia = 0; for (let i = 0; i < X.length; i++) inertia += dist2(X[i], C[labels[i]]);
      if (!best || inertia < best.inertia) best = { C, labels, inertia };
    }
    return { kind: "clusters", points: X.map((p, i) => ({ x: p[0], y: p[1], c: best.labels[i] })), centers: best.C.map(c => ({ x: c[0], y: c[1] })), centerType: "x", metric: { key: "inertia", val: best.inertia.toFixed(3) } };
  }

  function kmedoids(k) {
    const { X } = dataset("blobs3"), n = X.length, r = rng(808);
    let med = kmeansppInit(X, k, r).map(c => { let bi = 0, bd = Infinity; X.forEach((p, i) => { const d = dist2(p, c); if (d < bd) { bd = d; bi = i; } }); return bi; });
    const assign = () => X.map(p => { let bd = Infinity, bc = 0; med.forEach((m, c) => { const d = dist2(p, X[m]); if (d < bd) { bd = d; bc = c; } }); return bc; });
    const cost = m => { let s = 0; for (let i = 0; i < n; i++) { let bd = Infinity; m.forEach(mm => { const d = dist2(X[i], X[mm]); if (d < bd) bd = d; }); s += Math.sqrt(bd); } return s; };
    let cur = cost(med);
    for (let it = 0; it < 30; it++) {
      let improved = false;
      for (let c = 0; c < k; c++) for (let i = 0; i < n; i++) {
        if (med.includes(i)) continue; const trial = med.slice(); trial[c] = i; const tc = cost(trial);
        if (tc < cur - 1e-9) { med = trial; cur = tc; improved = true; }
      }
      if (!improved) break;
    }
    const labels = assign();
    return { kind: "clusters", points: X.map((p, i) => ({ x: p[0], y: p[1], c: labels[i] })), centers: med.map(m => ({ x: X[m][0], y: X[m][1] })), centerType: "ring", metric: { key: "coste", val: cur.toFixed(3) } };
  }

  function dbscan(eps, minPts) {
    const { X } = dataset("blobs3n"), n = X.length, e2 = eps * eps;
    const labels = new Array(n).fill(-2); // -2 undefined, -1 noise, >=0 cluster
    const region = i => { const out = []; for (let j = 0; j < n; j++) if (dist2(X[i], X[j]) <= e2) out.push(j); return out; };
    let cid = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -2) continue;
      const nb = region(i);
      if (nb.length < minPts) { labels[i] = -1; continue; }
      labels[i] = cid; const queue = nb.filter(j => j !== i);
      for (let q = 0; q < queue.length; q++) {
        const j = queue[q];
        if (labels[j] === -1) labels[j] = cid;
        if (labels[j] !== -2) continue;
        labels[j] = cid; const nb2 = region(j);
        if (nb2.length >= minPts) nb2.forEach(x => { if (queue.indexOf(x) === -1) queue.push(x); });
      }
      cid++;
    }
    const noise = labels.filter(l => l === -1).length;
    return { kind: "clusters", points: X.map((p, i) => ({ x: p[0], y: p[1], c: labels[i] })), centers: [], metric: { key: "clústeres", val: String(cid) }, metric2: { key: "ruido", val: String(noise) } };
  }

  function pca(nComp) {
    const { X } = dataset("cloud"), n = X.length;
    const mean = [0, 0]; X.forEach(p => { mean[0] += p[0]; mean[1] += p[1]; }); mean[0] /= n; mean[1] /= n;
    let a = 0, b = 0, c = 0; X.forEach(p => { const dx = p[0] - mean[0], dy = p[1] - mean[1]; a += dx * dx; b += dx * dy; c += dy * dy; });
    a /= n; b /= n; c /= n;
    const tr = a + c, det = a * c - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    let v1 = Math.abs(b) > 1e-9 ? [l1 - c, b] : [1, 0]; const nrm = Math.hypot(v1[0], v1[1]) || 1; v1 = [v1[0] / nrm, v1[1] / nrm];
    const v2 = [-v1[1], v1[0]];
    const s1 = Math.sqrt(l1) * 1.4, s2 = Math.sqrt(l2) * 1.4;
    const axes = [
      { x0: mean[0] - v1[0] * s1, y0: mean[1] - v1[1] * s1, x1: mean[0] + v1[0] * s1, y1: mean[1] + v1[1] * s1, main: true },
      { x0: mean[0] - v2[0] * s2, y0: mean[1] - v2[1] * s2, x1: mean[0] + v2[0] * s2, y1: mean[1] + v2[1] * s2, main: false }
    ];
    const points = X.map(p => ({ x: p[0], y: p[1] }));
    let proj = null;
    if (nComp === 1) proj = X.map(p => { const t = (p[0] - mean[0]) * v1[0] + (p[1] - mean[1]) * v1[1]; return { x: mean[0] + t * v1[0], y: mean[1] + t * v1[1], ox: p[0], oy: p[1] }; });
    const ev = l1 / (l1 + l2 || 1);
    return { kind: "pca", points, axes, proj, metric: { key: "var. PC1", val: (ev * 100).toFixed(1) + "%" }, metric2: { key: "var. PC2", val: ((1 - ev) * 100).toFixed(1) + "%" } };
  }

  // ============ NUEVOS PLAYGROUNDS ============

  // LightGBM — histogram-binned, leaf-wise gradient boosting (2nd-order, logistic loss).
  function lightgbm(numLeaves, lr, nEst, minChild) {
    const { X, y } = dataset("moonsn"), n = X.length, nFeat = X[0].length, maxBin = 32, lam = 1e-3;
    const edges = []; for (let f = 0; f < nFeat; f++) { const e = []; for (let b = 1; b < maxBin; b++) e.push(b / maxBin); edges.push(e); }
    const sig = z => 1 / (1 + Math.exp(-z));
    function leafwiseTree(g, hess) {
      const agg = idx => { let G = 0, H = 0; for (const i of idx) { G += g[i]; H += hess[i]; } return { G, H }; };
      const weight = (G, H) => -G / (H + lam);
      function bestSplit(idx) {
        if (idx.length < 2 * minChild) return null;
        const { G, H } = agg(idx), N = idx.length, parent = G * G / (H + lam); let best = null;
        for (let f = 0; f < nFeat; f++) for (const thr of edges[f]) {
          let GL = 0, HL = 0, nl = 0; for (const i of idx) if (X[i][f] <= thr) { GL += g[i]; HL += hess[i]; nl++; }
          const nr = N - nl; if (nl < minChild || nr < minChild) continue;
          const GR = G - GL, HR = H - HL, gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent;
          if (gain > 1e-9 && (!best || gain > best.gain)) best = { gain, f, thr };
        }
        return best;
      }
      const idx0 = X.map((_, i) => i), r0 = agg(idx0);
      const root = { idx: idx0, val: weight(r0.G, r0.H) }; root.split = bestSplit(idx0);
      const leaves = [root]; let count = 1;
      while (count < numLeaves) {
        let bi = -1, bg = 0; for (let k = 0; k < leaves.length; k++) if (leaves[k].split && leaves[k].split.gain > bg) { bg = leaves[k].split.gain; bi = k; }
        if (bi < 0) break;
        const nd = leaves[bi], sp = nd.split, L = [], R = []; for (const i of nd.idx) (X[i][sp.f] <= sp.thr ? L : R).push(i);
        nd.f = sp.f; nd.thr = sp.thr; nd.split = null;
        const rl = agg(L), rr = agg(R);
        nd.left = { idx: L, val: weight(rl.G, rl.H) }; nd.left.split = bestSplit(L);
        nd.right = { idx: R, val: weight(rr.G, rr.H) }; nd.right.split = bestSplit(R);
        leaves.splice(bi, 1, nd.left, nd.right); count++;
      }
      return p => { let nd = root; while (nd.left) nd = p[nd.f] <= nd.thr ? nd.left : nd.right; return nd.val; };
    }
    const pbar = y.reduce((s, v) => s + v, 0) / n, F0 = Math.log(pbar / (1 - pbar)), trees = [];
    const F = X.map(() => F0);
    for (let m = 0; m < nEst; m++) {
      const g = new Array(n), hess = new Array(n);
      for (let i = 0; i < n; i++) { const p = sig(F[i]); g[i] = p - y[i]; hess[i] = p * (1 - p); }
      const t = leafwiseTree(g, hess); trees.push(t);
      for (let i = 0; i < n; i++) F[i] += lr * t(X[i]);
    }
    const predict = p => { let v = F0; for (const t of trees) v += lr * t(p); return sig(v) > 0.5 ? 1 : 0; };
    return { kind: "regions", points: X.map((row, i) => ({ x: row[0], y: row[1], c: y[i] })), grid: classGrid(predict), metric: { key: "accuracy", val: accuracy(X, y, predict).toFixed(3) } };
  }

  // XGBoost (regression, squared-error) — 2nd-order regularized tree (Chen & Guestrin 2016).
  function xgbTree(X, g, h, maxDepth, lambda, gamma) {
    const nFeat = X[0].length;
    const leafW = idx => { let G = 0, H = 0; for (const i of idx) { G += g[i]; H += h[i]; } return -G / (H + lambda); };
    function grow(idx, depth) {
      if (depth >= maxDepth || idx.length < 2) return { leaf: true, w: leafW(idx) };
      let G = 0, H = 0; for (const i of idx) { G += g[i]; H += h[i]; }
      const base = G * G / (H + lambda); let best = null;
      for (let f = 0; f < nFeat; f++) {
        const vals = idx.map(i => X[i][f]).sort((a, b) => a - b), cands = [];
        for (let k = 0; k < vals.length - 1; k++) if (vals[k] !== vals[k + 1]) cands.push((vals[k] + vals[k + 1]) / 2);
        for (const thr of cands) {
          let GL = 0, HL = 0; for (const i of idx) if (X[i][f] <= thr) { GL += g[i]; HL += h[i]; }
          const GR = G - GL, HR = H - HL; if (HL <= 0 || HR <= 0) continue;
          const gain = 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) - base) - gamma;
          if (gain > 0 && (!best || gain > best.gain)) best = { gain, f, thr };
        }
      }
      if (!best) return { leaf: true, w: leafW(idx) };
      const L = [], R = []; for (const i of idx) (X[i][best.f] <= best.thr ? L : R).push(i);
      return { leaf: false, f: best.f, thr: best.thr, left: grow(L, depth + 1), right: grow(R, depth + 1) };
    }
    const root = grow(X.map((_, i) => i), 0);
    return p => { let n = root; while (!n.leaf) n = p[n.f] <= n.thr ? n.left : n.right; return n.w; };
  }
  function xgboost(eta, maxDepth, lambda, gamma) {
    const { X, y } = dataset("reg1d"), n = X.length, nEst = 16;
    const base = y.reduce((s, v) => s + v, 0) / n, trees = [];
    let F = X.map(() => base);
    for (let m = 0; m < nEst; m++) {
      const g = new Array(n), h = new Array(n);
      for (let i = 0; i < n; i++) { g[i] = F[i] - y[i]; h[i] = 1; }
      const t = xgbTree(X, g, h, maxDepth, lambda, gamma);
      trees.push(t); for (let i = 0; i < n; i++) F[i] += eta * t(X[i]);
    }
    const predict = x => { let v = base; trees.forEach(t => v += eta * t([x])); return v; };
    let mse = 0; for (let i = 0; i < n; i++) mse += (y[i] - F[i]) ** 2; mse /= n;
    let ssTot = 0; for (let i = 0; i < n; i++) ssTot += (y[i] - base) ** 2;
    const curve = []; for (let i = 0; i <= 80; i++) { const x = i / 80; curve.push({ x, y: predict(x) }); }
    return { kind: "curve", points: X.map((row, i) => ({ x: row[0], y: y[i] })), curve, metric: { key: "train MSE", val: mse.toFixed(4) }, metric2: { key: "R²", val: (ssTot ? 1 - mse * n / ssTot : 0).toFixed(3) } };
  }

  // HDBSCAN* (simplified but faithful): core dist -> mutual reachability MST -> condensed tree -> EOM/leaf.
  function hdbscan(minClusterSize, minSamples, selEps, selMethod) {
    const { X } = dataset("vardens"), n = X.length;
    const mcs = Math.max(2, minClusterSize | 0);
    const ms = Math.max(1, (minSamples | 0) || mcs);
    const D = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const d = Math.sqrt(dist2(X[i], X[j])); D[i][j] = d; D[j][i] = d; }
    const core = new Float64Array(n);
    for (let i = 0; i < n; i++) { const row = Array.from(D[i]).sort((a, b) => a - b); core[i] = row[Math.min(ms, n - 1)]; }
    const mreach = (i, j) => Math.max(core[i], core[j], D[i][j]);
    const inTree = new Uint8Array(n), best = new Float64Array(n).fill(Infinity), from = new Int32Array(n).fill(-1);
    best[0] = 0; const edges = [];
    for (let it = 0; it < n; it++) {
      let u = -1, bw = Infinity;
      for (let v = 0; v < n; v++) if (!inTree[v] && best[v] < bw) { bw = best[v]; u = v; }
      inTree[u] = 1;
      if (from[u] >= 0) edges.push({ a: from[u], b: u, w: bw });
      for (let v = 0; v < n; v++) if (!inTree[v]) { const w = mreach(u, v); if (w < best[v]) { best[v] = w; from[v] = u; } }
    }
    edges.sort((e1, e2) => e1.w - e2.w);
    const M = 2 * n - 1, parentUF = new Int32Array(M).fill(-1), nodeId = new Int32Array(n);
    for (let i = 0; i < n; i++) nodeId[i] = i;
    const find = x => { while (parentUF[x] >= 0) x = parentUF[x]; return x; };
    const size = new Int32Array(M), childL = new Int32Array(M).fill(-1), childR = new Int32Array(M).fill(-1), lamBorn = new Float64Array(M);
    for (let i = 0; i < n; i++) size[i] = 1;
    let next = n;
    for (const e of edges) {
      const ra = find(e.a), rb = find(e.b), na = nodeId[ra], nb = nodeId[rb], id = next++;
      childL[id] = na; childR[id] = nb; size[id] = size[na] + size[nb]; lamBorn[id] = e.w > 0 ? 1 / e.w : 1e9;
      parentUF[ra] = rb; nodeId[find(ra)] = id;
    }
    const root = 2 * n - 2;
    const relabel = new Int32Array(M).fill(-1), condParent = [], condBirthLam = [], stability = [];
    let nextLabel = 0;
    relabel[root] = nextLabel++; condParent[0] = -1; condBirthLam[0] = 0; stability[0] = 0;
    const memberCluster = new Int32Array(n).fill(-1);
    const pointsUnder = (node, cb) => { const st = [node]; while (st.length) { const x = st.pop(); if (x < n) cb(x); else { st.push(childL[x]); st.push(childR[x]); } } };
    const visit = [root];
    while (visit.length) {
      const node = visit.pop(); if (node < n) continue;
      const c = relabel[node], L = childL[node], R = childR[node], lam = lamBorn[node];
      const bigL = size[L] >= mcs, bigR = size[R] >= mcs;
      if (bigL && bigR) {
        [L, R].forEach(ch => { const nc = nextLabel++; relabel[ch] = nc; condParent[nc] = c; condBirthLam[nc] = lam; stability[nc] = 0; visit.push(ch); });
        stability[c] += (size[L] + size[R]) * (lam - condBirthLam[c]);
      } else {
        const stay = [], fall = []; (bigL ? stay : fall).push(L); (bigR ? stay : fall).push(R);
        let fc = 0; fall.forEach(fn => pointsUnder(fn, p => { fc++; memberCluster[p] = c; }));
        stability[c] += fc * (lam - condBirthLam[c]);
        stay.forEach(sn => { relabel[sn] = c; if (sn >= n) visit.push(sn); });
      }
    }
    const K = nextLabel, children = Array.from({ length: K }, () => []);
    for (let c = 1; c < K; c++) children[condParent[c]].push(c);
    let selected = new Uint8Array(K); const stabHat = stability.slice();
    if (selMethod === "leaf") {
      for (let c = 1; c < K; c++) if (children[c].length === 0) selected[c] = 1;
    } else {
      for (let c = K - 1; c >= 1; c--) {
        if (children[c].length === 0) { selected[c] = 1; continue; }
        const sub = children[c].reduce((s, ch) => s + stabHat[ch], 0);
        if (stability[c] >= sub) { selected[c] = 1; const st = [...children[c]]; while (st.length) { const d = st.pop(); selected[d] = 0; children[d].forEach(x => st.push(x)); } stabHat[c] = stability[c]; }
        else stabHat[c] = sub;
      }
    }
    if (selEps > 0) {
      const ns = new Uint8Array(K);
      for (let c = 1; c < K; c++) if (selected[c]) { let cur = c; while (condParent[cur] > 0 && (1 / condBirthLam[cur]) < selEps) cur = condParent[cur]; ns[cur] = 1; }
      for (let c = 1; c < K; c++) if (ns[c]) { let p = condParent[c]; while (p > 0) { if (ns[p]) { ns[c] = 0; break; } p = condParent[p]; } }
      selected = ns;
    }
    const sel = []; for (let c = 0; c < K; c++) if (selected[c]) sel.push(c);
    const colorOf = {}; sel.forEach((c, i) => colorOf[c] = i);
    const labels = new Array(n).fill(-1);
    for (let p = 0; p < n; p++) { let c = memberCluster[p]; while (c > 0) { if (selected[c]) { labels[p] = colorOf[c]; break; } c = condParent[c]; } }
    const noise = labels.filter(l => l < 0).length;
    return { kind: "clusters", points: X.map((pt, i) => ({ x: pt[0], y: pt[1], c: labels[i] })), centers: [], metric: { key: "clústeres", val: String(sel.length) }, metric2: { key: "ruido", val: String(noise) } };
  }

  // t-SNE (van der Maaten & Hinton 2008): P gaussiana calibrada a perplexity, Q t-Student, descenso de KL.
  function tsne(perplexity, learningRate, nIter) {
    const { X, y } = dataset("moons"), D = standardize(X).Z, n = D.length, dim = D[0].length;
    const dist = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let s = 0; for (let d = 0; d < dim; d++) { const df = D[i][d] - D[j][d]; s += df * df; }
      dist[i][j] = s; dist[j][i] = s;
    }
    const logU = Math.log(perplexity), P = Array.from({ length: n }, () => new Float64Array(n)), row = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let betaMin = -Infinity, betaMax = Infinity, beta = 1, sumP = 0;
      for (let tries = 0; tries < 60; tries++) {
        sumP = 0; for (let j = 0; j < n; j++) { if (j === i) { row[j] = 0; continue; } const v = Math.exp(-dist[i][j] * beta); row[j] = v; sumP += v; }
        sumP = sumP || 1e-12;
        let Hh = 0; for (let j = 0; j < n; j++) { const p = row[j] / sumP; if (p > 1e-12) Hh -= p * Math.log(p); }
        const diff = Hh - logU; if (Math.abs(diff) < 1e-5) break;
        if (diff > 0) { betaMin = beta; beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2; }
        else { betaMax = beta; beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2; }
      }
      for (let j = 0; j < n; j++) P[i][j] = row[j] / sumP;
    }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = (P[i][j] + P[j][i]) / (2 * n); P[i][j] = v; P[j][i] = v; }
    for (let i = 0; i < n; i++) P[i][i] = 0;
    const r = rng(2024), Y = Array.from({ length: n }, () => [gauss(r) * 1e-2, gauss(r) * 1e-2]), vel = Array.from({ length: n }, () => [0, 0]);
    const num = Array.from({ length: n }, () => new Float64Array(n)), earlyIters = Math.min(100, Math.floor(nIter * 0.3));
    for (let it = 0; it < nIter; it++) {
      const exag = it < earlyIters ? 12 : 1, mom = it < 50 ? 0.5 : 0.8;
      let qsum = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], w = 1 / (1 + dx * dx + dy * dy); num[i][j] = w; num[j][i] = w; qsum += 2 * w; }
      qsum = qsum || 1e-12;
      for (let i = 0; i < n; i++) {
        let gx = 0, gy = 0;
        for (let j = 0; j < n; j++) { if (i === j) continue; const mul = 4 * (P[i][j] * exag - num[i][j] / qsum) * num[i][j]; gx += mul * (Y[i][0] - Y[j][0]); gy += mul * (Y[i][1] - Y[j][1]); }
        vel[i][0] = mom * vel[i][0] - learningRate * gx; vel[i][1] = mom * vel[i][1] - learningRate * gy;
      }
      let mx = 0, my = 0; for (let i = 0; i < n; i++) { Y[i][0] += vel[i][0]; Y[i][1] += vel[i][1]; mx += Y[i][0]; my += Y[i][1]; }
      mx /= n; my /= n; for (let i = 0; i < n; i++) { Y[i][0] -= mx; Y[i][1] -= my; }
    }
    let qsum2 = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], w = 1 / (1 + dx * dx + dy * dy); num[i][j] = w; num[j][i] = w; qsum2 += 2 * w; }
    qsum2 = qsum2 || 1e-12; let kl = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { if (i === j) continue; const p = P[i][j]; if (p > 1e-12) kl += p * Math.log(p / Math.max(num[i][j] / qsum2, 1e-12)); }
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (let i = 0; i < n; i++) { if (!isFinite(Y[i][0])) Y[i][0] = 0; if (!isFinite(Y[i][1])) Y[i][1] = 0; minx = Math.min(minx, Y[i][0]); maxx = Math.max(maxx, Y[i][0]); miny = Math.min(miny, Y[i][1]); maxy = Math.max(maxy, Y[i][1]); }
    const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2, s = Math.max(maxx - minx, maxy - miny) || 1;
    const points = Y.map((p, i) => ({ x: 0.5 + 0.84 * (p[0] - cx) / s, y: 0.5 + 0.84 * (p[1] - cy) / s, c: y[i] }));
    return { kind: "clusters", points, centers: [], metric: { key: "KL", val: kl.toFixed(3) } };
  }

  // UMAP: conjunto simplicial difuso + SGD de entropía cruzada con muestreo negativo (real, determinista).
  function umap(nNeighbors, minDist) {
    function fitAB(md, spread) {
      const xs = [], ys = [];
      for (let x = 0.001; x <= 3; x += 0.03) { xs.push(x); ys.push(x <= md ? 1 : Math.exp(-(x - md) / spread)); }
      let a = 1, b = 1;
      for (let it = 0; it < 400; it++) {
        let ga = 0, gb = 0;
        for (let i = 0; i < xs.length; i++) {
          const x2b = Math.pow(xs[i], 2 * b), den = 1 + a * x2b, f = 1 / den, e = f - ys[i];
          ga += 2 * e * (-x2b / (den * den));
          gb += 2 * e * (-a * x2b * 2 * Math.log(xs[i]) / (den * den));
        }
        a -= 0.1 * ga / xs.length; b -= 0.1 * gb / xs.length;
        if (a < 1e-3) a = 1e-3; if (b < 1e-3) b = 1e-3;
      }
      return { a, b };
    }
    function pcaInit(X, D, n) {
      const mean = new Array(D).fill(0);
      X.forEach(row => row.forEach((v, j) => mean[j] += v)); mean.forEach((_, j) => mean[j] /= n);
      const Xc = X.map(row => row.map((v, j) => v - mean[j]));
      const C = Array.from({ length: D }, () => new Array(D).fill(0));
      Xc.forEach(row => { for (let a = 0; a < D; a++) for (let c = 0; c < D; c++) C[a][c] += row[a] * row[c]; });
      for (let a = 0; a < D; a++) for (let c = 0; c < D; c++) C[a][c] /= n;
      const power = M => {
        let v = new Array(D).fill(0).map((_, j) => Math.sin(j + 1));
        for (let it = 0; it < 100; it++) {
          const nv = new Array(D).fill(0);
          for (let a = 0; a < D; a++) for (let c = 0; c < D; c++) nv[a] += M[a][c] * v[c];
          const nrm = Math.hypot.apply(null, nv) || 1; v = nv.map(x => x / nrm);
        }
        let lam = 0; const Mv = new Array(D).fill(0);
        for (let a = 0; a < D; a++) for (let c = 0; c < D; c++) Mv[a] += M[a][c] * v[c];
        for (let a = 0; a < D; a++) lam += v[a] * Mv[a];
        return { v, lam };
      };
      const e1 = power(C);
      const C2 = C.map((row, a) => row.map((val, c) => val - e1.lam * e1.v[a] * e1.v[c]));
      const e2 = power(C2);
      const proj = Xc.map(row => { let p0 = 0, p1 = 0; for (let j = 0; j < D; j++) { p0 += row[j] * e1.v[j]; p1 += row[j] * e2.v[j]; } return [p0, p1]; });
      let m0 = 0; proj.forEach(p => m0 += p[0]); m0 /= n;
      let s0 = 0; proj.forEach(p => s0 += (p[0] - m0) ** 2); s0 = Math.sqrt(s0 / n) || 1;
      return proj.map(p => [p[0] / s0 * 4, p[1] / s0 * 4]);
    }
    const D = 6, perC = 30, K = 3;
    const r = rng(24601), Xh = [], lab = [];
    for (let c = 0; c < K; c++) {
      const ctr = [], dir = []; let nn = 0;
      for (let j = 0; j < D; j++) { ctr.push(gauss(r) * 1.6); const g = gauss(r); dir.push(g); nn += g * g; }
      nn = Math.sqrt(nn) || 1; for (let j = 0; j < D; j++) dir[j] /= nn;
      for (let i = 0; i < perC; i++) {
        const t = (i / (perC - 1) - 0.5) * 2.4, row = [];
        for (let j = 0; j < D; j++) row.push(ctr[j] + dir[j] * t + gauss(r) * 0.3);
        Xh.push(row); lab.push(c);
      }
    }
    const n = Xh.length;
    const dist = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let s = 0; for (let d = 0; d < D; d++) { const dd = Xh[i][d] - Xh[j][d]; s += dd * dd; }
      const v = Math.sqrt(s); dist[i][j] = v; dist[j][i] = v;
    }
    const k = Math.max(2, Math.min(n - 1, nNeighbors | 0)), targetLog = Math.log2(k);
    const neigh = [], Pd = {};
    for (let i = 0; i < n; i++) {
      const order = []; for (let j = 0; j < n; j++) if (j !== i) order.push(j);
      order.sort((a, b) => dist[i][a] - dist[i][b]);
      const kn = order.slice(0, k); neigh.push(kn);
      const rho = dist[i][kn[0]];
      let lo = 0, hi = 1e6, sigma = 1;
      for (let it = 0; it < 64; it++) {
        sigma = (lo + hi) / 2; let s = 0;
        for (let t = 0; t < kn.length; t++) { const dd = dist[i][kn[t]] - rho; s += dd > 0 ? Math.exp(-dd / sigma) : 1; }
        if (s > targetLog) hi = sigma; else lo = sigma;
      }
      for (let t = 0; t < kn.length; t++) { const dd = dist[i][kn[t]] - rho; Pd[i + '_' + kn[t]] = dd > 0 ? Math.exp(-dd / sigma) : 1; }
    }
    const W = {};
    for (const key in Pd) {
      const s = key.split('_'), i = +s[0], j = +s[1];
      const a = Pd[key] || 0, b = Pd[j + '_' + i] || 0, w = a + b - a * b;
      const uk = i < j ? i + '_' + j : j + '_' + i;
      if (w > 1e-3) W[uk] = w;
    }
    const edges = Object.keys(W).map(kk => { const s = kk.split('_'); return [+s[0], +s[1], W[kk]]; });
    const ab = fitAB(minDist, 1.0), A = ab.a, B = ab.b;
    const Y = pcaInit(Xh, D, n);
    const rr = rng(97531), epochs = 500, negRate = 5, clamp = g => g > 4 ? 4 : g < -4 ? -4 : g;
    for (let ep = 0; ep < epochs; ep++) {
      const alpha = 1.0 * (1 - ep / epochs);
      for (let e = 0; e < edges.length; e++) {
        const i = edges[e][0], j = edges[e][1], w = edges[e][2];
        if (rr() > w) continue;
        let dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], ds = dx * dx + dy * dy;
        const co = ds > 0 ? (-2 * A * B * Math.pow(ds, B - 1)) / (1 + A * Math.pow(ds, B)) : 0;
        const gx = clamp(co * dx) * alpha, gy = clamp(co * dy) * alpha;
        Y[i][0] += gx; Y[i][1] += gy; Y[j][0] -= gx; Y[j][1] -= gy;
        for (let s = 0; s < negRate; s++) {
          const kx = (rr() * n) | 0; if (kx === i) continue;
          dx = Y[i][0] - Y[kx][0]; dy = Y[i][1] - Y[kx][1]; ds = dx * dx + dy * dy;
          const cr = ds > 0 ? (2 * B) / ((0.001 + ds) * (1 + A * Math.pow(ds, B))) : 0;
          Y[i][0] += clamp(cr * dx) * alpha; Y[i][1] += clamp(cr * dy) * alpha;
        }
      }
    }
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    Y.forEach(p => { if (p[0] < mnX) mnX = p[0]; if (p[0] > mxX) mxX = p[0]; if (p[1] < mnY) mnY = p[1]; if (p[1] > mxY) mxY = p[1]; });
    const rawRange = Math.max(mxX - mnX, mxY - mnY);
    const sc = Math.max(34, rawRange / 0.92), cx = (mnX + mxX) / 2, cy = (mnY + mxY) / 2;
    const pts = Y.map((p, i) => ({ x: clip01(0.5 + (p[0] - cx) / sc), y: clip01(0.5 + (p[1] - cy) / sc), c: lab[i] }));
    let kept = 0, tot = 0;
    for (let i = 0; i < n; i++) {
      const lo = []; for (let j = 0; j < n; j++) if (j !== i) lo.push(j);
      lo.sort((u, v) => ((pts[i].x - pts[u].x) ** 2 + (pts[i].y - pts[u].y) ** 2) - ((pts[i].x - pts[v].x) ** 2 + (pts[i].y - pts[v].y) ** 2));
      const low = new Set(lo.slice(0, k));
      neigh[i].forEach(nj => { tot++; if (low.has(nj)) kept++; });
    }
    return { kind: "clusters", points: pts, centers: [], metric: { key: "vecinos conserv.", val: (tot ? kept / tot : 0).toFixed(2) }, metric2: { key: "k efectivo", val: String(k) } };
  }

  // ARIMA(p,d,q): estimación Hannan-Rissanen (OLS cerrado) sobre serie 1-D + pronóstico integrado.
  function arima(p, d, q) {
    p = Math.max(0, Math.min(4, Math.round(p)));
    d = Math.max(0, Math.min(2, Math.round(d)));
    q = Math.max(0, Math.min(2, Math.round(q)));
    const N = 48, r = rng(2718), y = [];
    let z1 = 0, z2 = 0;
    for (let i = 0; i < N; i++) {
      const z = 0.62 * z1 - 0.30 * z2 + gauss(r) * 0.05;
      z2 = z1; z1 = z;
      y.push(clip01(0.30 + 0.0075 * i + z));
    }
    const xAt = i => i / (N - 1);
    const split = Math.floor(N * 0.68), tr = y.slice(0, split), H = N - split;
    const levels = [tr.slice()];
    for (let k = 1; k <= d; k++) {
      const prev = levels[k - 1], cur = [];
      for (let i = 1; i < prev.length; i++) cur.push(prev[i] - prev[i - 1]);
      levels.push(cur);
    }
    const w = levels[d], m = w.length;
    function ols(A, target) {
      const cols = A[0].length, XtX = Array.from({ length: cols }, () => Array(cols).fill(0)), Xty = Array(cols).fill(0);
      for (let i = 0; i < A.length; i++) for (let a = 0; a < cols; a++) { Xty[a] += A[i][a] * target[i]; for (let c = 0; c < cols; c++) XtX[a][c] += A[i][a] * A[i][c]; }
      for (let a = 1; a < cols; a++) XtX[a][a] += 1e-4;
      return solve(XtX, Xty);
    }
    let ehat = new Array(m).fill(0);
    if (q > 0 && m > 6) {
      const k = Math.min(Math.max(p + q, 4) + 1, Math.floor(m / 2) - 1);
      if (k >= 1) {
        const A = [], t = [];
        for (let i = k; i < m; i++) { const row = [1]; for (let l = 1; l <= k; l++) row.push(w[i - l]); A.push(row); t.push(w[i]); }
        const bAR = ols(A, t);
        for (let i = k; i < m; i++) { let pr = bAR[0]; for (let l = 1; l <= k; l++) pr += bAR[l] * w[i - l]; ehat[i] = w[i] - pr; }
      }
    }
    const start = Math.max(p, q, 1);
    let c, phi = [], theta = [];
    if ((p === 0 && q === 0) || m - start < p + q + 2) {
      c = w.reduce((s, v) => s + v, 0) / (m || 1);
    } else {
      const A = [], t = [];
      for (let i = start; i < m; i++) {
        const row = [1];
        for (let l = 1; l <= p; l++) row.push(w[i - l]);
        for (let l = 1; l <= q; l++) row.push(ehat[i - l]);
        A.push(row); t.push(w[i]);
      }
      const coef = ols(A, t);
      c = coef[0];
      for (let l = 1; l <= p; l++) phi.push(coef[l] || 0);
      for (let l = 1; l <= q; l++) theta.push(coef[p + l] || 0);
    }
    if (!isFinite(c)) c = w.reduce((s, v) => s + v, 0) / (m || 1);
    phi = phi.map(v => isFinite(v) ? v : 0);
    theta = theta.map(v => isFinite(v) ? v : 0);
    function run(phi, theta) {
      const e = new Array(m).fill(0), predW = new Array(m).fill(0);
      let rss = 0, cnt = 0;
      for (let i = start; i < m; i++) {
        let pr = c;
        for (let l = 1; l <= p; l++) pr += phi[l - 1] * w[i - l];
        for (let l = 1; l <= q; l++) pr += theta[l - 1] * e[i - l];
        predW[i] = pr; e[i] = w[i] - pr; rss += e[i] * e[i]; cnt++;
      }
      const wf = [], getW = idx => idx < m ? w[idx] : wf[idx - m], getE = idx => idx < m ? e[idx] : 0;
      for (let h = 1; h <= H; h++) {
        const idx = m - 1 + h; let pr = c;
        for (let l = 1; l <= p; l++) pr += phi[l - 1] * getW(idx - l);
        for (let l = 1; l <= q; l++) pr += theta[l - 1] * getE(idx - l);
        wf.push(pr);
      }
      let fc = wf.slice();
      for (let k = d - 1; k >= 0; k--) {
        const anchor = levels[k][levels[k].length - 1];
        let running = anchor; const up = [];
        for (let h = 0; h < fc.length; h++) { running += fc[h]; up.push(running); }
        fc = up;
      }
      return { predW, fc, rss, cnt };
    }
    let damp = 1, R;
    for (let g = 0; g < 16; g++) {
      R = run(phi.map(v => v * damp), theta.map(v => v * damp));
      if (R.fc.every(v => v > -0.6 && v < 1.6)) break;
      damp *= 0.8;
    }
    const binom = (dd, kk) => { let cc = 1; for (let i = 0; i < kk; i++) cc = cc * (dd - i) / (i + 1); return cc; };
    const curve = [];
    for (let i = start; i < m; i++) {
      const t = i + d; let lvl = R.predW[i];
      for (let k = 1; k <= d; k++) lvl += ((k % 2 === 1) ? 1 : -1) * binom(d, k) * tr[t - k];
      curve.push({ x: xAt(t), y: lvl });
    }
    for (let h = 1; h <= H; h++) curve.push({ x: xAt(split - 1 + h), y: R.fc[h - 1] });
    let se = 0; for (let h = 1; h <= H; h++) { const err = R.fc[h - 1] - y[split - 1 + h]; se += err * err; }
    const rmse = Math.sqrt(se / H);
    const aic = R.cnt > 0 ? R.cnt * Math.log(R.rss / R.cnt + 1e-12) + 2 * (p + q + 1) : 0;
    return { kind: "curve", points: y.map((v, i) => ({ x: xAt(i), y: v })), curve, metric: { key: "RMSE fc", val: rmse.toFixed(3) }, metric2: { key: "AIC", val: aic.toFixed(1) } };
  }

  // Prophet: y = g(t) [+|*] s(t). MAP = mínimos cuadrados penalizados (L1 changepoints, L2 Fourier), prox-GD.
  function prophet(cps, sps, mode, cpRange) {
    const { X, y } = dataset("prophetTs"), n = X.length, t = X.map(r => r[0]);
    const mult = mode === "multiplicative";
    const S = 15, cpt = []; for (let j = 0; j < S; j++) cpt.push(cpRange * (j + 1) / (S + 1));
    const H = 4, P = 0.25;
    const cosF = [], sinF = [];
    for (let i = 0; i < n; i++) { const c = [], s = []; for (let h = 1; h <= H; h++) { const w = 2 * Math.PI * h * t[i] / P; c.push(Math.cos(w)); s.push(Math.sin(w)); } cosF.push(c); sinF.push(s); }
    let m = 0, k = 0; const delta = Array(S).fill(0), a = Array(H).fill(0), b = Array(H).fill(0);
    const trendAt = (i, dl) => { let g = m + k * t[i]; for (let j = 0; j < S; j++) { const d = t[i] - cpt[j]; if (d > 0) g += dl[j] * d; } return g; };
    const seasAt = (i) => { let s = 0; for (let h = 0; h < H; h++) s += a[h] * cosF[i][h] + b[h] * sinF[i][h]; return s; };
    const lamCp = 0.0002 / cps, lamSeas = 0.5 / (sps * sps);
    const lr = 0.4, iters = 2500, thr = lr * lamCp;
    const soft = (v, th) => v > th ? v - th : (v < -th ? v + th : 0);
    for (let it = 0; it < iters; it++) {
      let gm = 0, gk = 0; const gd = Array(S).fill(0), ga = Array(H).fill(0), gb = Array(H).fill(0);
      for (let i = 0; i < n; i++) {
        const g = trendAt(i, delta), s = seasAt(i);
        const pred = mult ? g * (1 + s) : g + s, e = pred - y[i];
        const dT = mult ? (1 + s) : 1, dS = mult ? g : 1;
        gm += e * dT; gk += e * dT * t[i];
        for (let j = 0; j < S; j++) { const d = t[i] - cpt[j]; if (d > 0) gd[j] += e * dT * d; }
        for (let h = 0; h < H; h++) { ga[h] += e * dS * cosF[i][h]; gb[h] += e * dS * sinF[i][h]; }
      }
      m -= lr * gm / n; k -= lr * gk / n;
      for (let j = 0; j < S; j++) delta[j] = soft(delta[j] - lr * gd[j] / n, thr);
      for (let h = 0; h < H; h++) { a[h] = (a[h] - lr * ga[h] / n) / (1 + 2 * lr * lamSeas); b[h] = (b[h] - lr * gb[h] / n) / (1 + 2 * lr * lamSeas); }
    }
    const predAt = tt => {
      let g = m + k * tt; for (let j = 0; j < S; j++) { const d = tt - cpt[j]; if (d > 0) g += delta[j] * d; }
      let s = 0; for (let h = 1; h <= H; h++) { const w = 2 * Math.PI * h * tt / P; s += a[h - 1] * Math.cos(w) + b[h - 1] * Math.sin(w); }
      return mult ? g * (1 + s) : g + s;
    };
    let sse = 0; for (let i = 0; i < n; i++) sse += (predAt(t[i]) - y[i]) ** 2;
    const active = delta.filter(d => Math.abs(d) > 1e-3).length;
    const curve = []; for (let i = 0; i <= 80; i++) { const x = i / 80; curve.push({ x, y: predAt(x) }); }
    return { kind: "curve", points: X.map((r, i) => ({ x: r[0], y: y[i] })), curve, metric: { key: "RMSE", val: Math.sqrt(sse / n).toFixed(4) }, metric2: { key: "changepoints", val: String(active) } };
  }

  // LSTM (1 capa, BPTT real, regresión secuencial 1-D). Determinista (semilla fija + máscara dropout sembrada).
  function lstm(units, epochs, dropout) {
    const H = units;
    const T = 48, rs = rng(4242), tArr = [], v = [];
    for (let k = 0; k < T; k++) {
      const tk = k / (T - 1);
      const val = 0.5 + 0.26 * Math.sin(2 * Math.PI * 1.2 * tk) + 0.14 * Math.sin(2 * Math.PI * 3.3 * tk + 0.9) + gauss(rs) * 0.02;
      tArr.push(tk); v.push(clip01(val));
    }
    let mean = 0; for (let k = 0; k < T; k++) mean += v[k]; mean /= T;
    let sd = 0; for (let k = 0; k < T; k++) sd += (v[k] - mean) ** 2; sd = Math.sqrt(sd / T) || 1;
    const z = v.map(x => (x - mean) / sd);
    const N = T - 1;
    const U = z.slice(0, N), Y = z.slice(1, T);
    const r = rng(20240722), sx = 0.6, sh = 0.6 / Math.sqrt(H);
    const v1 = () => Array(H).fill(0);
    const v2 = () => Array.from({ length: H }, () => Array(H).fill(0));
    const Wxi = [], Wxf = [], Wxo = [], Wxg = [];
    for (let j = 0; j < H; j++) { Wxi.push(gauss(r) * sx); Wxf.push(gauss(r) * sx); Wxo.push(gauss(r) * sx); Wxg.push(gauss(r) * sx); }
    const Whi = v2(), Whf = v2(), Who = v2(), Whg = v2();
    for (let j = 0; j < H; j++) for (let m = 0; m < H; m++) { Whi[j][m] = gauss(r) * sh; Whf[j][m] = gauss(r) * sh; Who[j][m] = gauss(r) * sh; Whg[j][m] = gauss(r) * sh; }
    const bi = v1(), bf = Array(H).fill(1), bo = v1(), bg = v1();
    const Vr = Array.from({ length: H }, () => gauss(r) * 0.3); let by = 0;
    const sig = x => 1 / (1 + Math.exp(-x));
    const mWxi = v1(), mWxf = v1(), mWxo = v1(), mWxg = v1();
    const mWhi = v2(), mWhf = v2(), mWho = v2(), mWhg = v2();
    const mbi = v1(), mbf = v1(), mbo = v1(), mbg = v1();
    const mVr = v1(); let mby = 0;
    const lr = 0.05, mom = 0.9, clip = 5;
    function forward(mask) {
      const cache = [], preds = new Array(N);
      let hPrev = v1(), csPrev = v1();
      for (let k = 0; k < N; k++) {
        const u = U[k];
        const hp = mask ? hPrev.map((h, idx) => h * mask[idx]) : hPrev;
        const iG = v1(), fG = v1(), oG = v1(), gG = v1(), cs = v1(), tc = v1(), h = v1();
        for (let j = 0; j < H; j++) {
          let ai = Wxi[j] * u + bi[j], af = Wxf[j] * u + bf[j], ao = Wxo[j] * u + bo[j], ag = Wxg[j] * u + bg[j];
          for (let m = 0; m < H; m++) { ai += Whi[j][m] * hp[m]; af += Whf[j][m] * hp[m]; ao += Who[j][m] * hp[m]; ag += Whg[j][m] * hp[m]; }
          iG[j] = sig(ai); fG[j] = sig(af); oG[j] = sig(ao); gG[j] = Math.tanh(ag);
          cs[j] = fG[j] * csPrev[j] + iG[j] * gG[j];
          tc[j] = Math.tanh(cs[j]);
          h[j] = oG[j] * tc[j];
        }
        let yh = by; for (let j = 0; j < H; j++) yh += Vr[j] * h[j];
        preds[k] = yh;
        cache.push({ u, hp, csPrev, iG, fG, oG, gG, cs, tc, h });
        hPrev = h; csPrev = cs;
      }
      return { cache, preds };
    }
    for (let ep = 0; ep < epochs; ep++) {
      let mask = null;
      if (dropout > 0) { const rm = rng(9001 + ep); mask = Array.from({ length: H }, () => rm() < dropout ? 0 : 1 / (1 - dropout)); }
      const { cache, preds } = forward(mask);
      const gWxi = v1(), gWxf = v1(), gWxo = v1(), gWxg = v1();
      const gWhi = v2(), gWhf = v2(), gWho = v2(), gWhg = v2();
      const gbi = v1(), gbf = v1(), gbo = v1(), gbg = v1();
      const gVr = v1(); let gby = 0;
      let dhNext = v1(), dcsNext = v1();
      for (let k = N - 1; k >= 0; k--) {
        const c = cache[k], dy = (preds[k] - Y[k]) / N;
        gby += dy;
        const dh = v1();
        for (let j = 0; j < H; j++) { gVr[j] += dy * c.h[j]; dh[j] = dy * Vr[j] + dhNext[j]; }
        const dcsPrev = v1(), dhPrev = v1();
        for (let j = 0; j < H; j++) {
          const dao = (dh[j] * c.tc[j]) * c.oG[j] * (1 - c.oG[j]);
          const dcs = dcsNext[j] + dh[j] * c.oG[j] * (1 - c.tc[j] * c.tc[j]);
          const dai = (dcs * c.gG[j]) * c.iG[j] * (1 - c.iG[j]);
          const daf = (dcs * c.csPrev[j]) * c.fG[j] * (1 - c.fG[j]);
          const dag = (dcs * c.iG[j]) * (1 - c.gG[j] * c.gG[j]);
          dcsPrev[j] = dcs * c.fG[j];
          gWxi[j] += dai * c.u; gWxf[j] += daf * c.u; gWxo[j] += dao * c.u; gWxg[j] += dag * c.u;
          gbi[j] += dai; gbf[j] += daf; gbo[j] += dao; gbg[j] += dag;
          for (let m = 0; m < H; m++) {
            gWhi[j][m] += dai * c.hp[m]; gWhf[j][m] += daf * c.hp[m]; gWho[j][m] += dao * c.hp[m]; gWhg[j][m] += dag * c.hp[m];
            const mm = mask ? mask[m] : 1;
            dhPrev[m] += (Whi[j][m] * dai + Whf[j][m] * daf + Who[j][m] * dao + Whg[j][m] * dag) * mm;
          }
        }
        dhNext = dhPrev; dcsNext = dcsPrev;
      }
      let nrm = gby * gby;
      for (let j = 0; j < H; j++) {
        nrm += gVr[j] ** 2 + gWxi[j] ** 2 + gWxf[j] ** 2 + gWxo[j] ** 2 + gWxg[j] ** 2 + gbi[j] ** 2 + gbf[j] ** 2 + gbo[j] ** 2 + gbg[j] ** 2;
        for (let m = 0; m < H; m++) nrm += gWhi[j][m] ** 2 + gWhf[j][m] ** 2 + gWho[j][m] ** 2 + gWhg[j][m] ** 2;
      }
      nrm = Math.sqrt(nrm); const sc = nrm > clip ? clip / nrm : 1;
      const up1 = (w, g, mv) => { for (let j = 0; j < H; j++) { mv[j] = mom * mv[j] - lr * g[j] * sc; w[j] += mv[j]; } };
      const up2 = (w, g, mv) => { for (let j = 0; j < H; j++) for (let m = 0; m < H; m++) { mv[j][m] = mom * mv[j][m] - lr * g[j][m] * sc; w[j][m] += mv[j][m]; } };
      up1(Wxi, gWxi, mWxi); up1(Wxf, gWxf, mWxf); up1(Wxo, gWxo, mWxo); up1(Wxg, gWxg, mWxg);
      up2(Whi, gWhi, mWhi); up2(Whf, gWhf, mWhf); up2(Who, gWho, mWho); up2(Whg, gWhg, mWhg);
      up1(bi, gbi, mbi); up1(bf, gbf, mbf); up1(bo, gbo, mbo); up1(bg, gbg, mbg);
      for (let j = 0; j < H; j++) { mVr[j] = mom * mVr[j] - lr * gVr[j] * sc; Vr[j] += mVr[j]; }
      mby = mom * mby - lr * gby * sc; by += mby;
    }
    const { preds } = forward(null);
    const curve = [];
    for (let k = 0; k < N; k++) curve.push({ x: tArr[k + 1], y: preds[k] * sd + mean });
    let ybar = 0; for (let k = 0; k < N; k++) ybar += v[k + 1]; ybar /= N;
    let mse = 0, ssTot = 0;
    for (let k = 0; k < N; k++) { const yhat = preds[k] * sd + mean; mse += (yhat - v[k + 1]) ** 2; ssTot += (v[k + 1] - ybar) ** 2; }
    mse /= N;
    const r2 = ssTot ? 1 - (mse * N) / ssTot : 0;
    const points = tArr.map((tk, k) => ({ x: tk, y: v[k] }));
    return { kind: "curve", points, curve, metric: { key: "train MSE", val: mse.toFixed(4) }, metric2: { key: "R²", val: r2.toFixed(3) } };
  }

  // Autoencoder (2D -> latente L -> 2D, backprop real, MSE) sobre la nube "arc". Reutiliza renderKind 'pca'.
  function autoencoder(latentDim, hidden, activation) {
    const { X } = dataset("arc");
    const S = standardize(X), Z = S.Z, n = Z.length, r = rng(4242);
    const L = Math.max(1, Math.min(2, Math.round(latentDim)));
    const H = Math.max(1, Math.round(hidden));
    const linear = activation === "linear", relu = activation === "relu";
    const act = linear ? (x => x) : relu ? (x => Math.max(0, x)) : Math.tanh;
    const dact = linear ? (_ => 1) : relu ? (a => a > 0 ? 1 : 0) : (a => 1 - a * a);
    const rnd = () => gauss(r) * 0.5;
    let W1 = Array.from({ length: H }, () => [rnd(), rnd()]), b1 = Array(H).fill(0);
    let Wz = Array.from({ length: L }, () => Array.from({ length: H }, rnd)), bz = Array(L).fill(0);
    let Wd = Array.from({ length: H }, () => Array.from({ length: L }, rnd)), bd = Array(H).fill(0);
    let Wo = Array.from({ length: 2 }, () => Array.from({ length: H }, rnd)), bo = [0, 0];
    const lr = 0.15, epochs = 900;
    const fwd = z => {
      const h1 = new Array(H); for (let h = 0; h < H; h++) h1[h] = act(b1[h] + W1[h][0] * z[0] + W1[h][1] * z[1]);
      const zc = new Array(L); for (let l = 0; l < L; l++) { let s = bz[l]; for (let h = 0; h < H; h++) s += Wz[l][h] * h1[h]; zc[l] = s; }
      const h2 = new Array(H); for (let h = 0; h < H; h++) { let s = bd[h]; for (let l = 0; l < L; l++) s += Wd[h][l] * zc[l]; h2[h] = act(s); }
      const out = [bo[0], bo[1]]; for (let o = 0; o < 2; o++) for (let h = 0; h < H; h++) out[o] += Wo[o][h] * h2[h];
      return { h1, zc, h2, out };
    };
    for (let ep = 0; ep < epochs; ep++) {
      const gW1 = W1.map(() => [0, 0]), gb1 = Array(H).fill(0);
      const gWz = Wz.map(() => Array(H).fill(0)), gbz = Array(L).fill(0);
      const gWd = Wd.map(() => Array(L).fill(0)), gbd = Array(H).fill(0);
      const gWo = Wo.map(() => Array(H).fill(0)), gbo = [0, 0];
      for (let i = 0; i < n; i++) {
        const z = Z[i], f = fwd(z), eo = [f.out[0] - z[0], f.out[1] - z[1]];
        const dh2 = Array(H).fill(0);
        for (let o = 0; o < 2; o++) { gbo[o] += eo[o]; for (let h = 0; h < H; h++) { gWo[o][h] += eo[o] * f.h2[h]; dh2[h] += eo[o] * Wo[o][h]; } }
        const dpre2 = Array(H); for (let h = 0; h < H; h++) dpre2[h] = dh2[h] * dact(f.h2[h]);
        const dz = Array(L).fill(0);
        for (let h = 0; h < H; h++) { gbd[h] += dpre2[h]; for (let l = 0; l < L; l++) { gWd[h][l] += dpre2[h] * f.zc[l]; dz[l] += dpre2[h] * Wd[h][l]; } }
        const dh1 = Array(H).fill(0);
        for (let l = 0; l < L; l++) { gbz[l] += dz[l]; for (let h = 0; h < H; h++) { gWz[l][h] += dz[l] * f.h1[h]; dh1[h] += dz[l] * Wz[l][h]; } }
        for (let h = 0; h < H; h++) { const dpre1 = dh1[h] * dact(f.h1[h]); gb1[h] += dpre1; gW1[h][0] += dpre1 * z[0]; gW1[h][1] += dpre1 * z[1]; }
      }
      const s = lr / n;
      for (let h = 0; h < H; h++) { b1[h] -= s * gb1[h]; W1[h][0] -= s * gW1[h][0]; W1[h][1] -= s * gW1[h][1]; bd[h] -= s * gbd[h]; for (let l = 0; l < L; l++) Wd[h][l] -= s * gWd[h][l]; }
      for (let l = 0; l < L; l++) { bz[l] -= s * gbz[l]; for (let h = 0; h < H; h++) Wz[l][h] -= s * gWz[l][h]; }
      for (let o = 0; o < 2; o++) { bo[o] -= s * gbo[o]; for (let h = 0; h < H; h++) Wo[o][h] -= s * gWo[o][h]; }
    }
    const destd = v => [v[0] * S.std[0] + S.mean[0], v[1] * S.std[1] + S.mean[1]];
    const decode = zc => {
      const h2 = new Array(H); for (let h = 0; h < H; h++) { let s = bd[h]; for (let l = 0; l < L; l++) s += Wd[h][l] * zc[l]; h2[h] = act(s); }
      const out = [bo[0], bo[1]]; for (let o = 0; o < 2; o++) for (let h = 0; h < H; h++) out[o] += Wo[o][h] * h2[h];
      return destd(out);
    };
    const proj = X.map((p, i) => { const rec = destd(fwd(Z[i]).out); return { ox: p[0], oy: p[1], x: rec[0], y: rec[1] }; });
    let mse = 0; proj.forEach(q => { mse += (q.ox - q.x) ** 2 + (q.oy - q.y) ** 2; }); mse /= n;
    const axes = [];
    if (L === 1) {
      const zs = X.map((_, i) => fwd(Z[i]).zc[0]);
      const zmin = Math.min.apply(null, zs), zmax = Math.max.apply(null, zs), pad = (zmax - zmin) * 0.04 || 0.1;
      let prev = null;
      for (let i = 0; i <= 60; i++) { const t = zmin - pad + (zmax - zmin + 2 * pad) * i / 60; const q = decode([t]); if (prev) axes.push({ x0: prev[0], y0: prev[1], x1: q[0], y1: q[1], main: true }); prev = q; }
    }
    return { kind: "pca", points: X.map(p => ({ x: p[0], y: p[1] })), axes, proj, metric: { key: "MSE recon", val: mse.toFixed(4) }, metric2: { key: "latente", val: L + "D" } };
  }

  // β-VAE sobre reg1d reinterpretado como puntos 2-D en una variedad 1-D. ELBO por descenso de gradiente.
  function vae(beta, lr, epochs) {
    const D0 = dataset("reg1d"), X = D0.X.map((row, i) => [row[0], D0.y[i]]);
    const n = X.length, D = 2, He = 6, L = 1, Hd = 6, RW = 15;
    const r = rng(1717), gi = s => gauss(r) * s, sig = z => 1 / (1 + Math.exp(-z));
    const cE = v => Math.max(-8, Math.min(4, v));
    const M = (a, b, s) => Array.from({ length: a }, () => Array.from({ length: b }, () => gi(s)));
    const Z2 = (a, b) => Array.from({ length: a }, () => Array(b).fill(0));
    let We = M(He, D, 0.7), be = Array(He).fill(0), Wmu = M(L, He, 0.5), bmu = Array(L).fill(0),
        Wlv = M(L, He, 0.5), blv = Array(L).fill(0), Wd = M(Hd, L, 0.7), bd = Array(Hd).fill(0),
        Wo = M(D, Hd, 0.7), bo = Array(D).fill(0);
    const enc = x => {
      const h = new Array(He);
      for (let j = 0; j < He; j++) { let a = be[j]; for (let d = 0; d < D; d++) a += We[j][d] * x[d]; h[j] = Math.tanh(a); }
      const mu = new Array(L), lv = new Array(L);
      for (let l = 0; l < L; l++) { let am = bmu[l], al = blv[l]; for (let j = 0; j < He; j++) { am += Wmu[l][j] * h[j]; al += Wlv[l][j] * h[j]; } mu[l] = Math.max(-10, Math.min(10, am)); lv[l] = cE(al); }
      return { h, mu, lv };
    };
    const dec = z => {
      const h = new Array(Hd);
      for (let j = 0; j < Hd; j++) { let a = bd[j]; for (let l = 0; l < L; l++) a += Wd[j][l] * z[l]; h[j] = Math.tanh(a); }
      const o = new Array(D);
      for (let d = 0; d < D; d++) { let a = bo[d]; for (let j = 0; j < Hd; j++) a += Wo[d][j] * h[j]; o[d] = sig(a); }
      return { h, o };
    };
    for (let ep = 0; ep < epochs; ep++) {
      const gWe = Z2(He, D), gbe = Array(He).fill(0), gWmu = Z2(L, He), gbmu = Array(L).fill(0),
            gWlv = Z2(L, He), gblv = Array(L).fill(0), gWd = Z2(Hd, L), gbd = Array(Hd).fill(0),
            gWo = Z2(D, Hd), gbo = Array(D).fill(0);
      for (let s = 0; s < n; s++) {
        const x = X[s], E = enc(x), he = E.h, mu = E.mu, lv = E.lv;
        const eps = new Array(L), z = new Array(L), std = new Array(L);
        for (let l = 0; l < L; l++) { eps[l] = gauss(r); std[l] = Math.exp(0.5 * lv[l]); z[l] = mu[l] + std[l] * eps[l]; }
        const Dc = dec(z), hd = Dc.h, xhat = Dc.o, doo = new Array(D);
        for (let d = 0; d < D; d++) { doo[d] = RW * (xhat[d] - x[d]); gbo[d] += doo[d]; for (let j = 0; j < Hd; j++) gWo[d][j] += doo[d] * hd[j]; }
        const dhd = new Array(Hd).fill(0);
        for (let j = 0; j < Hd; j++) for (let d = 0; d < D; d++) dhd[j] += doo[d] * Wo[d][j];
        const dhp = new Array(Hd), dz = new Array(L).fill(0);
        for (let j = 0; j < Hd; j++) { dhp[j] = dhd[j] * (1 - hd[j] * hd[j]); gbd[j] += dhp[j]; for (let l = 0; l < L; l++) { gWd[j][l] += dhp[j] * z[l]; dz[l] += dhp[j] * Wd[j][l]; } }
        const dmu = new Array(L), dlv = new Array(L), dhe = new Array(He).fill(0);
        for (let l = 0; l < L; l++) {
          dmu[l] = dz[l] + beta * mu[l];
          dlv[l] = dz[l] * (std[l] * eps[l]) * 0.5 + beta * 0.5 * (Math.exp(lv[l]) - 1);
          gbmu[l] += dmu[l]; gblv[l] += dlv[l];
          for (let j = 0; j < He; j++) { gWmu[l][j] += dmu[l] * he[j]; gWlv[l][j] += dlv[l] * he[j]; dhe[j] += dmu[l] * Wmu[l][j] + dlv[l] * Wlv[l][j]; }
        }
        for (let j = 0; j < He; j++) { const dp = dhe[j] * (1 - he[j] * he[j]); gbe[j] += dp; for (let d = 0; d < D; d++) gWe[j][d] += dp * x[d]; }
      }
      const up = g => Math.max(-6, Math.min(6, g / n)) * lr;
      for (let j = 0; j < He; j++) { be[j] -= up(gbe[j]); for (let d = 0; d < D; d++) We[j][d] -= up(gWe[j][d]); }
      for (let l = 0; l < L; l++) { bmu[l] = Math.max(-10, Math.min(10, bmu[l] - up(gbmu[l]))); blv[l] = cE(blv[l] - up(gblv[l])); for (let j = 0; j < He; j++) { Wmu[l][j] -= up(gWmu[l][j]); Wlv[l][j] -= up(gWlv[l][j]); } }
      for (let j = 0; j < Hd; j++) { bd[j] -= up(gbd[j]); for (let l = 0; l < L; l++) Wd[j][l] -= up(gWd[j][l]); }
      for (let d = 0; d < D; d++) { bo[d] -= up(gbo[d]); for (let j = 0; j < Hd; j++) Wo[d][j] -= up(gWo[d][j]); }
    }
    let mse = 0, kl = 0;
    for (let s = 0; s < n; s++) { const E = enc(X[s]), o = dec(E.mu).o; mse += (o[0] - X[s][0]) ** 2 + (o[1] - X[s][1]) ** 2; for (let l = 0; l < L; l++) kl += 0.5 * (E.mu[l] * E.mu[l] + Math.exp(E.lv[l]) - 1 - E.lv[l]); }
    mse /= n; kl /= n;
    const curve = [];
    for (let i = 0; i <= 80; i++) { const zz = -2.6 + 5.2 * (i / 80), o = dec([zz]).o; curve.push({ x: o[0], y: o[1] }); }
    return { kind: "curve", points: X.map(p => ({ x: p[0], y: p[1] })), curve, metric: { key: "recon MSE", val: mse.toFixed(4) }, metric2: { key: "KL", val: kl.toFixed(3) } };
  }

  // PINN: red tanh 1-capa u(x) entrenada con L = L_datos + lambda·L_PDE. EDO u'(x)=A·cos(Wx), anclaje u(0)=0.5.
  function pinn(lambda, nColloc, hidden) {
    const W = 2 * Math.PI * 1.4, AMP = 0.30, A = AMP * W;
    const utrue = x => 0.5 + AMP * Math.sin(W * x);
    const rd = rng(717), xd = [0], yd = [0.5];
    [0.30, 0.42, 0.50, 0.58, 0.70].forEach(x => { xd.push(x); yd.push(clip01(utrue(x) + gauss(rd) * 0.045)); });
    const Nd = xd.length;
    const xc = []; if (nColloc <= 1) xc.push(0.5); else for (let k = 0; k < nColloc; k++) xc.push(k / (nColloc - 1));
    const Nc = xc.length, H = hidden, r = rng(4242);
    let w = Array.from({ length: H }, () => gauss(r) * 6.0);
    let b = Array.from({ length: H }, () => gauss(r) * 3.0);
    let v = Array.from({ length: H }, () => gauss(r) * 0.30), c = 0.5;
    const fwd = x => { let u = c, ux = 0; const a = new Array(H), fp = new Array(H), fpp = new Array(H);
      for (let h = 0; h < H; h++) { const zz = w[h] * x + b[h], t = Math.tanh(zz); a[h] = t; fp[h] = 1 - t * t; fpp[h] = -2 * t * (1 - t * t); u += v[h] * a[h]; ux += v[h] * fp[h] * w[h]; }
      return { u, ux, a, fp, fpp }; };
    const lr = 0.06, iters = 1400, GCLIP = 3.0;
    for (let it = 0; it < iters; it++) {
      const gw = new Array(H).fill(0), gb = new Array(H).fill(0), gv = new Array(H).fill(0); let gc = 0;
      for (let i = 0; i < Nd; i++) { const x = xd[i], f = fwd(x), e = (f.u - yd[i]) / Nd; gc += e;
        for (let h = 0; h < H; h++) { gv[h] += e * f.a[h]; gb[h] += e * v[h] * f.fp[h]; gw[h] += e * v[h] * f.fp[h] * x; } }
      for (let j = 0; j < Nc; j++) { const x = xc[j], f = fwd(x), res = f.ux - A * Math.cos(W * x), rr = lambda * res / Nc;
        for (let h = 0; h < H; h++) { gv[h] += rr * (f.fp[h] * w[h]); gb[h] += rr * (v[h] * w[h] * f.fpp[h]); gw[h] += rr * (v[h] * (f.fp[h] + w[h] * f.fpp[h] * x)); } }
      let nrm = gc * gc; for (let h = 0; h < H; h++) nrm += gw[h] * gw[h] + gb[h] * gb[h] + gv[h] * gv[h]; nrm = Math.sqrt(nrm);
      const s = nrm > GCLIP ? GCLIP / nrm : 1;
      for (let h = 0; h < H; h++) { w[h] -= lr * s * gw[h]; b[h] -= lr * s * gb[h]; v[h] -= lr * s * gv[h]; } c -= lr * s * gc;
    }
    let dmse = 0; for (let i = 0; i < Nd; i++) { const e = fwd(xd[i]).u - yd[i]; dmse += e * e; } dmse /= Nd;
    let pres = 0; const NG = 60; for (let k = 0; k <= NG; k++) { const x = k / NG, f = fwd(x), res = f.ux - A * Math.cos(W * x); pres += res * res; } pres /= (NG + 1);
    const curve = []; for (let i = 0; i <= 80; i++) { const x = i / 80; curve.push({ x, y: fwd(x).u }); }
    return { kind: "curve", points: xd.map((x, i) => ({ x, y: yd[i] })), curve, metric: { key: "residuo EDP", val: pres.toFixed(3) }, metric2: { key: "MSE datos", val: dmse.toFixed(4) } };
  }

  // ---- dispatch ----
  const FIT = {
    ridge: p => ridgePoly(p.alpha, p.degree, p.fit_intercept),
    logreg: p => logreg(p.C, p.penalty),
    gnb: p => gnb(Math.pow(10, p.var_exp)),
    svmlin: p => svmLinear(p.C),
    tree: p => tree(p.max_depth, p.min_samples_leaf, p.criterion),
    forest: p => forest(p.n_estimators, p.max_depth, p.max_features),
    gboost: p => gboost(p.n_estimators, p.learning_rate, p.max_depth),
    mlp: p => mlp(p.hidden_units, p.activation, p.learning_rate, p.epochs),
    kmeans: p => kmeans(p.k, p.n_init),
    kmedoids: p => kmedoids(p.k),
    dbscan: p => dbscan(p.eps, p.min_samples),
    pca: p => pca(p.n_components),
    lightgbm: p => lightgbm(p.num_leaves, p.learning_rate, p.n_estimators, p.min_child_samples),
    xgboost: p => xgboost(p.eta, p.max_depth, p.lambda, p.gamma),
    hdbscan: p => hdbscan(p.min_cluster_size, p.min_samples, p.cluster_selection_epsilon, p.cluster_selection_method),
    tsne: p => tsne(p.perplexity, p.learning_rate, p.n_iter),
    umap: p => umap(p.n_neighbors, p.min_dist),
    arima: p => arima(p.p, p.d, p.q),
    prophet: p => prophet(p.changepoint_prior_scale, p.seasonality_prior_scale, p.seasonality_mode, p.changepoint_range),
    lstm: p => lstm(p.units, p.epochs, p.dropout),
    autoencoder: p => autoencoder(p.latent_dim, p.hidden_units, p.activation),
    vae: p => vae(p.beta, p.learning_rate, p.epochs),
    pinn: p => pinn(p.lambda_pde, p.n_collocation, p.hidden_units)
  };
  window.MLKit = { fit: function (type, params) { return FIT[type](params); }, GRID };
})();
