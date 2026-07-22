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
    const { X, y } = dataset("blobs2"), n = X.length, d = 2, classes = [0, 1];
    let maxVar = 0; const dim = [0, 1];
    const stats = classes.map(c => {
      const rows = X.filter((_, i) => y[i] === c), mean = [0, 0], varr = [0, 0];
      rows.forEach(r => dim.forEach(j => mean[j] += r[j])); dim.forEach(j => mean[j] /= rows.length);
      rows.forEach(r => dim.forEach(j => varr[j] += (r[j] - mean[j]) ** 2)); dim.forEach(j => varr[j] /= rows.length);
      dim.forEach(j => { if (varr[j] > maxVar) maxVar = varr[j]; });
      return { c, mean, varr, prior: rows.length / n, rows: rows.length };
    });
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
    const cost = m => { let s = 0; for (let i = 0; i < n; i++) { let bd = Infinity; m.forEach(mm => { const d = dist2(X[i], X[mm]); if (d < bd) bd = d; }); s += bd; } return s; };
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
    pca: p => pca(p.n_components)
  };
  window.MLKit = { fit: function (type, params) { return FIT[type](params); }, GRID };
})();
