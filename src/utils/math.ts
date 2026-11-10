import { Centroid, DataDot } from '../types/ml';

// ----------------------------------------------------
// 1. K-MEANS CLUSTERING
// ----------------------------------------------------

export interface KMeansResult {
  centroids: Centroid[];
  dots: DataDot[];
  inertia: number;
  converged: boolean;
  iteration: number;
}

const CLUSTER_COLORS = [
  '#06b6d4', // cyan-500
  '#f59e0b', // amber-500
  '#8b5cf6', // violet-500
  '#10b981', // emerald-500
  '#f43f5e', // rose-500
  '#3b82f6', // blue-500
];

export function initCentroids(dots: DataDot[], k: number, mode: 'kmeans++' | 'random'): Centroid[] {
  if (dots.length === 0) return [];

  const centroids: Centroid[] = [];

  if (mode === 'random' || dots.length < k) {
    // Pick random dots as initial centroids
    const indices = new Set<number>();
    while (indices.size < Math.min(k, dots.length)) {
      indices.add(Math.floor(Math.random() * dots.length));
    }
    let id = 0;
    indices.forEach((idx) => {
      centroids.push({
        id,
        x: dots[idx].x,
        y: dots[idx].y,
        color: CLUSTER_COLORS[id % CLUSTER_COLORS.length],
      });
      id++;
    });
    return centroids;
  }

  // K-Means++ Initialization
  // 1. Pick first centroid randomly
  const firstIdx = Math.floor(Math.random() * dots.length);
  centroids.push({
    id: 0,
    x: dots[firstIdx].x,
    y: dots[firstIdx].y,
    color: CLUSTER_COLORS[0],
  });

  // 2. Pick remaining centroids with probability proportional to D(x)^2
  for (let c = 1; c < k; c++) {
    const distancesSq = dots.map((d) => {
      let minDistSq = Infinity;
      for (const cent of centroids) {
        const distSq = (d.x - cent.x) ** 2 + (d.y - cent.y) ** 2;
        if (distSq < minDistSq) minDistSq = distSq;
      }
      return minDistSq;
    });

    const sumDistSq = distancesSq.reduce((a, b) => a + b, 0);
    let r = Math.random() * sumDistSq;
    let chosenIdx = 0;
    for (let i = 0; i < dots.length; i++) {
      r -= distancesSq[i];
      if (r <= 0) {
        chosenIdx = i;
        break;
      }
    }

    centroids.push({
      id: c,
      x: dots[chosenIdx].x,
      y: dots[chosenIdx].y,
      color: CLUSTER_COLORS[c % CLUSTER_COLORS.length],
    });
  }

  return centroids;
}

export function runKMeansStep(dots: DataDot[], centroids: Centroid[]): KMeansResult {
  if (centroids.length === 0 || dots.length === 0) {
    return { centroids, dots, inertia: 0, converged: true, iteration: 0 };
  }

  // Step 1: Assign each dot to nearest centroid
  let inertia = 0;
  const updatedDots = dots.map((d) => {
    let bestDistSq = Infinity;
    let bestCluster = 0;
    centroids.forEach((c) => {
      const distSq = (d.x - c.x) ** 2 + (d.y - c.y) ** 2;
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        bestCluster = c.id;
      }
    });
    inertia += bestDistSq;
    return { ...d, cluster: bestCluster };
  });

  // Step 2: Recalculate centroid positions
  let maxShift = 0;
  const newCentroids = centroids.map((c) => {
    const assigned = updatedDots.filter((d) => d.cluster === c.id);
    if (assigned.length === 0) return c; // keep stationary if empty

    const avgX = assigned.reduce((acc, d) => acc + d.x, 0) / assigned.length;
    const avgY = assigned.reduce((acc, d) => acc + d.y, 0) / assigned.length;
    const shift = Math.hypot(avgX - c.x, avgY - c.y);
    if (shift > maxShift) maxShift = shift;

    return {
      ...c,
      x: avgX,
      y: avgY,
    };
  });

  return {
    centroids: newCentroids,
    dots: updatedDots,
    inertia: Number(inertia.toFixed(4)),
    converged: maxShift < 0.001,
    iteration: 1,
  };
}

// ----------------------------------------------------
// 2. LINEAR REGRESSION & GRADIENT DESCENT
// ----------------------------------------------------

export interface LinearRegressionResult {
  slope: number;
  intercept: number;
  mse: number;
  r2: number;
  dotsWithResiduals: DataDot[];
}

export function computeOLS(dots: DataDot[]): LinearRegressionResult {
  if (dots.length < 2) {
    return { slope: 0, intercept: 0.5, mse: 0, r2: 0, dotsWithResiduals: dots };
  }

  const n = dots.length;
  const sumX = dots.reduce((a, d) => a + d.x, 0);
  const sumY = dots.reduce((a, d) => a + d.y, 0);
  const meanX = sumX / n;
  const meanY = sumY / n;

  let num = 0;
  let den = 0;
  let totalSS = 0;

  dots.forEach((d) => {
    num += (d.x - meanX) * (d.y - meanY);
    den += (d.x - meanX) ** 2;
    totalSS += (d.y - meanY) ** 2;
  });

  const slope = den === 0 ? 0 : num / den;
  const intercept = meanY - slope * meanX;

  let residualSS = 0;
  const dotsWithResiduals = dots.map((d) => {
    const yHat = slope * d.x + intercept;
    const residual = d.y - yHat;
    residualSS += residual ** 2;
    return { ...d, residual };
  });

  const mse = residualSS / n;
  const r2 = totalSS === 0 ? 1 : Math.max(0, 1 - residualSS / totalSS);

  return {
    slope,
    intercept,
    mse,
    r2,
    dotsWithResiduals,
  };
}

export interface GradientDescentState {
  theta0: number; // intercept
  theta1: number; // slope
  loss: number;
  history: Array<{ theta0: number; theta1: number; loss: number }>;
}

export function stepGradientDescent(
  dots: DataDot[],
  current: GradientDescentState,
  learningRate: number
): GradientDescentState {
  const n = dots.length;
  if (n === 0) return current;

  let grad0 = 0;
  let grad1 = 0;
  let totalLoss = 0;

  dots.forEach((d) => {
    const pred = current.theta1 * d.x + current.theta0;
    const error = pred - d.y;
    grad0 += error;
    grad1 += error * d.x;
    totalLoss += error ** 2;
  });

  grad0 = (2 / n) * grad0;
  grad1 = (2 / n) * grad1;
  const loss = totalLoss / n;

  const nextTheta0 = current.theta0 - learningRate * grad0;
  const nextTheta1 = current.theta1 - learningRate * grad1;

  const newHistory = [...current.history, { theta0: nextTheta0, theta1: nextTheta1, loss }];
  if (newHistory.length > 50) newHistory.shift();

  return {
    theta0: nextTheta0,
    theta1: nextTheta1,
    loss,
    history: newHistory,
  };
}

// ----------------------------------------------------
// 3. K-NEAREST NEIGHBORS (KNN)
// ----------------------------------------------------

export interface KNNQueryOutcome {
  queryX: number;
  queryY: number;
  neighbors: Array<{ dot: DataDot; distance: number; weight: number }>;
  probabilities: { [label: number]: number };
  predictedLabel: number;
  decisionMargin: number;
}

export function queryKNN(
  dots: DataDot[],
  qx: number,
  qy: number,
  k: number,
  weighting: 'uniform' | 'distance'
): KNNQueryOutcome {
  if (dots.length === 0) {
    return {
      queryX: qx,
      queryY: qy,
      neighbors: [],
      probabilities: { 0: 0.5, 1: 0.5 },
      predictedLabel: 0,
      decisionMargin: 0,
    };
  }

  // Calculate distances to all dots
  const withDist = dots.map((d) => {
    const dist = Math.hypot(d.x - qx, d.y - qy);
    return { dot: d, distance: dist, weight: 1 };
  });

  // Sort ascending by distance
  withDist.sort((a, b) => a.distance - b.distance);
  const actualK = Math.min(k, dots.length);
  const kNearest = withDist.slice(0, actualK);

  // Compute weights
  let totalWeight = 0;
  const labelWeights: { [label: number]: number } = { 0: 0, 1: 0 };

  kNearest.forEach((item) => {
    const w = weighting === 'distance' ? 1 / Math.max(0.001, item.distance) : 1;
    item.weight = w;
    totalWeight += w;
    const lbl = item.dot.label ?? 0;
    labelWeights[lbl] = (labelWeights[lbl] || 0) + w;
  });

  const prob0 = totalWeight > 0 ? (labelWeights[0] || 0) / totalWeight : 0.5;
  const prob1 = totalWeight > 0 ? (labelWeights[1] || 0) / totalWeight : 0.5;
  const predictedLabel = prob1 >= prob0 ? 1 : 0;

  return {
    queryX: qx,
    queryY: qy,
    neighbors: kNearest,
    probabilities: { 0: prob0, 1: prob1 },
    predictedLabel,
    decisionMargin: Math.abs(prob1 - prob0),
  };
}

// ----------------------------------------------------
// 4. SUPPORT VECTOR MACHINE (SVM)
// ----------------------------------------------------

export interface SVMResult {
  weights: [number, number]; // w1, w2
  bias: number;
  marginWidth: number;
  supportVectors: DataDot[];
  kernel: 'linear' | 'rbf';
  accuracy: number;
}

export function solveSVM(
  dots: DataDot[],
  cParam: number,
  kernel: 'linear' | 'rbf',
  gamma = 5.0
): SVMResult {
  if (dots.length < 2) {
    return {
      weights: [0, 1],
      bias: -0.5,
      marginWidth: 0.2,
      supportVectors: [],
      kernel,
      accuracy: 100,
    };
  }

  // Simplified robust Linear SVM solver via Pegasos/Sub-gradient formulation
  let w1 = 0.5;
  let w2 = 0.5;
  let b = -0.5;
  const lr = 0.05;
  const epochs = 80;

  for (let ep = 0; ep < epochs; ep++) {
    dots.forEach((d) => {
      const y = (d.label ?? 0) === 1 ? 1 : -1;
      const score = y * (w1 * d.x + w2 * d.y + b);
      if (score < 1) {
        // Hinge loss violation
        w1 = w1 + lr * (cParam * y * d.x - 0.01 * w1);
        w2 = w2 + lr * (cParam * y * d.y - 0.01 * w2);
        b = b + lr * cParam * y;
      } else {
        w1 = w1 - lr * 0.01 * w1;
        w2 = w2 - lr * 0.01 * w2;
      }
    });
  }

  const normW = Math.hypot(w1, w2);
  const marginWidth = normW > 0 ? 1 / normW : 0.2;

  // Identify support vectors: dots closest to margin boundary or violating it
  let correct = 0;
  const supportVectors: DataDot[] = [];

  dots.forEach((d) => {
    const y = (d.label ?? 0) === 1 ? 1 : -1;
    const functionalMargin = y * (w1 * d.x + w2 * d.y + b);
    const isSV = functionalMargin <= 1.2;
    if (isSV) {
      supportVectors.push({ ...d, isSupportVector: true, alpha: Math.max(0, 1.2 - functionalMargin) });
    }
    if (functionalMargin > 0) correct++;
  });

  return {
    weights: [w1, w2],
    bias: b,
    marginWidth,
    supportVectors,
    kernel,
    accuracy: Math.round((correct / dots.length) * 100),
  };
}

// ----------------------------------------------------
// 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
// ----------------------------------------------------

export interface PCAResult {
  mean: { x: number; y: number };
  pc1: { x: number; y: number; eigenvalue: number };
  pc2: { x: number; y: number; eigenvalue: number };
  varianceExplained1: number;
  varianceExplained2: number;
  projectedDots: Array<DataDot & { projX: number; projY: number; score: number }>;
}

export function computePCA(dots: DataDot[]): PCAResult {
  if (dots.length < 2) {
    return {
      mean: { x: 0.5, y: 0.5 },
      pc1: { x: 1, y: 0, eigenvalue: 1 },
      pc2: { x: 0, y: 1, eigenvalue: 0 },
      varianceExplained1: 1,
      varianceExplained2: 0,
      projectedDots: dots.map((d) => ({ ...d, projX: d.x, projY: d.y, score: 0 })),
    };
  }

  const n = dots.length;
  const meanX = dots.reduce((a, d) => a + d.x, 0) / n;
  const meanY = dots.reduce((a, d) => a + d.y, 0) / n;

  // Covariance matrix components
  let varX = 0;
  let varY = 0;
  let covXY = 0;

  dots.forEach((d) => {
    const dx = d.x - meanX;
    const dy = d.y - meanY;
    varX += dx * dx;
    varY += dy * dy;
    covXY += dx * dy;
  });

  varX /= n - 1;
  varY /= n - 1;
  covXY /= n - 1;

  // Eigenvalues of 2x2 symmetric matrix: [[varX, covXY], [covXY, varY]]
  // trace = varX + varY, det = varX*varY - covXY^2
  const trace = varX + varY;
  const det = varX * varY - covXY * covXY;
  const disc = Math.max(0, (trace * trace) / 4 - det);
  const lambda1 = trace / 2 + Math.sqrt(disc);
  const lambda2 = Math.max(0, trace / 2 - Math.sqrt(disc));

  // Eigenvector for lambda1
  let e1x = 1;
  let e1y = 0;
  if (Math.abs(covXY) > 1e-6) {
    e1x = lambda1 - varY;
    e1y = covXY;
  } else if (varX >= varY) {
    e1x = 1;
    e1y = 0;
  } else {
    e1x = 0;
    e1y = 1;
  }
  const norm1 = Math.hypot(e1x, e1y) || 1;
  e1x /= norm1;
  e1y /= norm1;

  // PC2 is orthogonal to PC1
  const e2x = -e1y;
  const e2y = e1x;

  const totalVar = lambda1 + lambda2 || 1;
  const ve1 = lambda1 / totalVar;
  const ve2 = lambda2 / totalVar;

  // Orthogonal projection onto PC1 axis
  const projectedDots = dots.map((d) => {
    const dx = d.x - meanX;
    const dy = d.y - meanY;
    const score = dx * e1x + dy * e1y; // scalar projection
    const projX = meanX + score * e1x;
    const projY = meanY + score * e1y;
    return {
      ...d,
      projX,
      projY,
      score,
    };
  });

  return {
    mean: { x: meanX, y: meanY },
    pc1: { x: e1x, y: e1y, eigenvalue: lambda1 },
    pc2: { x: e2x, y: e2y, eigenvalue: lambda2 },
    varianceExplained1: ve1,
    varianceExplained2: ve2,
    projectedDots,
  };
}

// ----------------------------------------------------
// 6. DECISION TREE SPLITTER
// ----------------------------------------------------

export interface DecisionTreeSplit {
  feature: 'x' | 'y';
  threshold: number;
  leftClass: number;
  rightClass: number;
  impurity: number;
}

export interface TreePartition {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  predictedLabel: number;
  purity: number;
  count: number;
}

export function buildDecisionTreePartitions(
  dots: DataDot[],
  maxDepth: number,
  criterion: 'gini' | 'entropy'
): { partitions: TreePartition[]; splits: DecisionTreeSplit[] } {
  if (dots.length === 0) return { partitions: [], splits: [] };

  const splits: DecisionTreeSplit[] = [];

  function calculateImpurity(sample: DataDot[]): number {
    if (sample.length === 0) return 0;
    const p1 = sample.filter((d) => d.label === 1).length / sample.length;
    const p0 = 1 - p1;
    if (criterion === 'gini') {
      return 1 - (p0 * p0 + p1 * p1);
    } else {
      const e0 = p0 > 0 ? -p0 * Math.log2(p0) : 0;
      const e1 = p1 > 0 ? -p1 * Math.log2(p1) : 0;
      return e0 + e1;
    }
  }

  function splitNode(
    currentDots: DataDot[],
    depth: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number
  ): TreePartition[] {
    const count1 = currentDots.filter((d) => d.label === 1).length;
    const count0 = currentDots.length - count1;
    const majorityLabel = count1 >= count0 ? 1 : 0;
    const purity = currentDots.length > 0 ? Math.max(count0, count1) / currentDots.length : 1;

    // Base condition
    if (depth >= maxDepth || purity >= 0.95 || currentDots.length <= 4) {
      return [
        {
          xMin,
          xMax,
          yMin,
          yMax,
          predictedLabel: majorityLabel,
          purity,
          count: currentDots.length,
        },
      ];
    }

    // Evaluate potential split thresholds across X and Y
    let bestGain = -Infinity;
    let bestFeature: 'x' | 'y' = 'x';
    let bestThreshold = 0.5;
    const baseImpurity = calculateImpurity(currentDots);

    const candidates = [0.2, 0.35, 0.5, 0.65, 0.8];
    (['x', 'y'] as const).forEach((feat) => {
      candidates.forEach((th) => {
        const left = currentDots.filter((d) => (feat === 'x' ? d.x : d.y) <= th);
        const right = currentDots.filter((d) => (feat === 'x' ? d.x : d.y) > th);
        if (left.length === 0 || right.length === 0) return;

        const leftImp = calculateImpurity(left);
        const rightImp = calculateImpurity(right);
        const weightedImp = (left.length / currentDots.length) * leftImp + (right.length / currentDots.length) * rightImp;
        const gain = baseImpurity - weightedImp;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feat;
          bestThreshold = th;
        }
      });
    });

    if (bestGain <= 0.01) {
      return [
        {
          xMin,
          xMax,
          yMin,
          yMax,
          predictedLabel: majorityLabel,
          purity,
          count: currentDots.length,
        },
      ];
    }

    splits.push({
      feature: bestFeature,
      threshold: bestThreshold,
      leftClass: majorityLabel,
      rightClass: 1 - majorityLabel,
      impurity: baseImpurity - bestGain,
    });

    if (bestFeature === 'x') {
      const leftDots = currentDots.filter((d) => d.x <= bestThreshold);
      const rightDots = currentDots.filter((d) => d.x > bestThreshold);
      return [
        ...splitNode(leftDots, depth + 1, xMin, bestThreshold, yMin, yMax),
        ...splitNode(rightDots, depth + 1, bestThreshold, xMax, yMin, yMax),
      ];
    } else {
      const bottomDots = currentDots.filter((d) => d.y <= bestThreshold);
      const topDots = currentDots.filter((d) => d.y > bestThreshold);
      return [
        ...splitNode(bottomDots, depth + 1, xMin, xMax, yMin, bestThreshold),
        ...splitNode(topDots, depth + 1, xMin, xMax, bestThreshold, yMax),
      ];
    }
  }

  const partitions = splitNode(dots, 0, 0, 1, 0, 1);
  return { partitions, splits };
}

// ----------------------------------------------------
// 7. NEURAL NETWORK / MULTI-LAYER PERCEPTRON
// ----------------------------------------------------

export interface NeuralNetWeights {
  w1: number[][]; // [hiddenNeurons][2 inputs]
  b1: number[]; // [hiddenNeurons]
  w2: number[]; // [hiddenNeurons] output weights
  b2: number;
}

export function initNeuralNet(hiddenNeurons: number): NeuralNetWeights {
  const w1: number[][] = [];
  const b1: number[] = [];
  const w2: number[] = [];

  for (let i = 0; i < hiddenNeurons; i++) {
    w1.push([(Math.random() - 0.5) * 1.5, (Math.random() - 0.5) * 1.5]);
    b1.push((Math.random() - 0.5) * 0.5);
    w2.push((Math.random() - 0.5) * 1.5);
  }

  return {
    w1,
    b1,
    w2,
    b2: (Math.random() - 0.5) * 0.5,
  };
}

export function forwardPassNN(
  x1: number,
  x2: number,
  weights: NeuralNetWeights,
  activation: 'relu' | 'sigmoid' | 'tanh'
): { output: number; hiddenActivations: number[] } {
  const hiddenActivations: number[] = [];

  for (let i = 0; i < weights.w1.length; i++) {
    const z = weights.w1[i][0] * x1 + weights.w1[i][1] * x2 + weights.b1[i];
    let a = 0;
    if (activation === 'relu') {
      a = Math.max(0, z);
    } else if (activation === 'sigmoid') {
      a = 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, z))));
    } else {
      a = Math.tanh(z);
    }
    hiddenActivations.push(a);
  }

  let zOut = weights.b2;
  for (let i = 0; i < hiddenActivations.length; i++) {
    zOut += hiddenActivations[i] * weights.w2[i];
  }

  const output = 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, zOut))));
  return { output, hiddenActivations };
}

export function trainEpochNN(
  dots: DataDot[],
  weights: NeuralNetWeights,
  lr: number,
  activation: 'relu' | 'sigmoid' | 'tanh'
): { updatedWeights: NeuralNetWeights; loss: number } {
  let totalLoss = 0;
  const updated = {
    w1: weights.w1.map((row) => [...row]),
    b1: [...weights.b1],
    w2: [...weights.w2],
    b2: weights.b2,
  };

  dots.forEach((d) => {
    const target = d.label ?? 0;
    const { output, hiddenActivations } = forwardPassNN(d.x, d.y, updated, activation);
    const error = output - target;
    totalLoss += error * error;

    // Gradient of loss w.r.t zOut: error * sigmoid_derivative
    const deltaOut = error * output * (1 - output);

    // Update output layer weights
    for (let i = 0; i < hiddenActivations.length; i++) {
      updated.w2[i] -= lr * deltaOut * hiddenActivations[i];

      // Backpropagate to hidden layer
      let dAct = 1;
      const hAct = hiddenActivations[i];
      if (activation === 'relu') {
        dAct = hAct > 0 ? 1 : 0;
      } else if (activation === 'sigmoid') {
        dAct = hAct * (1 - hAct);
      } else {
        dAct = 1 - hAct * hAct;
      }

      const deltaHidden = deltaOut * weights.w2[i] * dAct;
      updated.w1[i][0] -= lr * deltaHidden * d.x;
      updated.w1[i][1] -= lr * deltaHidden * d.y;
      updated.b1[i] -= lr * deltaHidden;
    }
    updated.b2 -= lr * deltaOut;
  });

  return {
    updatedWeights: updated,
    loss: dots.length > 0 ? totalLoss / dots.length : 0,
  };
}

// ----------------------------------------------------
// 8. DBSCAN DENSITY CLUSTERING
// ----------------------------------------------------

export interface DBSCANResult {
  dots: DataDot[];
  numClusters: number;
  coreCount: number;
  borderCount: number;
  noiseCount: number;
  clusterColors: string[];
}

const DBSCAN_PALETTE = [
  '#14b8a6', // teal-500
  '#f59e0b', // amber-500
  '#8b5cf6', // violet-500
  '#06b6d4', // cyan-500
  '#ec4899', // pink-500
  '#3b82f6', // blue-500
  '#10b981', // emerald-500
];

export function runDBSCAN(dots: DataDot[], eps: number, minPts: number): DBSCANResult {
  const n = dots.length;
  if (n === 0) {
    return {
      dots: [],
      numClusters: 0,
      coreCount: 0,
      borderCount: 0,
      noiseCount: 0,
      clusterColors: DBSCAN_PALETTE,
    };
  }

  // Precompute neighbor indices for each dot
  const neighbors: number[][] = [];
  for (let i = 0; i < n; i++) {
    const list: number[] = [];
    for (let j = 0; j < n; j++) {
      if (Math.hypot(dots[i].x - dots[j].x, dots[i].y - dots[j].y) <= eps) {
        list.push(j);
      }
    }
    neighbors.push(list);
  }

  const isCore: boolean[] = neighbors.map((nbrs) => nbrs.length >= minPts);
  const clusterAssigned: number[] = new Array(n).fill(-1);
  let currentCluster = 0;

  for (let i = 0; i < n; i++) {
    if (clusterAssigned[i] !== -1 || !isCore[i]) continue;

    clusterAssigned[i] = currentCluster;
    const queue: number[] = [...neighbors[i]];

    while (queue.length > 0) {
      const qIdx = queue.shift()!;
      if (clusterAssigned[qIdx] === -1) {
        clusterAssigned[qIdx] = currentCluster;
        if (isCore[qIdx]) {
          queue.push(...neighbors[qIdx].filter((idx) => clusterAssigned[idx] === -1));
        }
      }
    }
    currentCluster++;
  }

  let coreCount = 0;
  let borderCount = 0;
  let noiseCount = 0;

  const updatedDots: DataDot[] = dots.map((d, i) => {
    let type: 'core' | 'border' | 'noise' = 'noise';
    if (clusterAssigned[i] !== -1) {
      if (isCore[i]) {
        type = 'core';
        coreCount++;
      } else {
        type = 'border';
        borderCount++;
      }
    } else {
      type = 'noise';
      noiseCount++;
    }

    return {
      ...d,
      cluster: clusterAssigned[i] !== -1 ? clusterAssigned[i] : undefined,
      dbscanCluster: clusterAssigned[i],
      dbscanType: type,
    };
  });

  return {
    dots: updatedDots,
    numClusters: currentCluster,
    coreCount,
    borderCount,
    noiseCount,
    clusterColors: DBSCAN_PALETTE,
  };
}

// ----------------------------------------------------
// 9. LOGISTIC REGRESSION
// ----------------------------------------------------

export interface LogisticRegressionResult {
  w1: number;
  w2: number;
  b: number;
  logLoss: number;
  accuracy: number;
  predictProb: (x: number, y: number) => number;
}

export function solveLogisticRegression(
  dots: DataDot[],
  learningRate = 0.1,
  epochs = 60,
  threshold = 0.5
): LogisticRegressionResult {
  let w1 = 0.5;
  let w2 = 0.5;
  let b = -0.5;
  const n = dots.length;

  if (n === 0) {
    return {
      w1: 0,
      w2: 0,
      b: 0,
      logLoss: 0,
      accuracy: 100,
      predictProb: () => 0.5,
    };
  }

  for (let ep = 0; ep < epochs; ep++) {
    let gradW1 = 0;
    let gradW2 = 0;
    let gradB = 0;

    dots.forEach((d) => {
      const target = d.label ?? 0;
      const z = w1 * d.x + w2 * d.y + b;
      const prob = 1 / (1 + Math.exp(-Math.max(-12, Math.min(12, z))));
      const error = prob - target;

      gradW1 += error * d.x;
      gradW2 += error * d.y;
      gradB += error;
    });

    w1 -= (learningRate / n) * gradW1;
    w2 -= (learningRate / n) * gradW2;
    b -= (learningRate / n) * gradB;
  }

  let totalLoss = 0;
  let correct = 0;

  dots.forEach((d) => {
    const target = d.label ?? 0;
    const z = w1 * d.x + w2 * d.y + b;
    const prob = 1 / (1 + Math.exp(-Math.max(-12, Math.min(12, z))));
    const clampedProb = Math.max(1e-7, Math.min(1 - 1e-7, prob));
    totalLoss += -(target * Math.log(clampedProb) + (1 - target) * Math.log(1 - clampedProb));

    const predClass = prob >= threshold ? 1 : 0;
    if (predClass === target) correct++;
  });

  const predictProb = (x: number, y: number) => {
    const z = w1 * x + w2 * y + b;
    return 1 / (1 + Math.exp(-Math.max(-12, Math.min(12, z))));
  };

  return {
    w1,
    w2,
    b,
    logLoss: totalLoss / n,
    accuracy: Math.round((correct / n) * 100),
    predictProb,
  };
}

// ----------------------------------------------------
// 10. GAUSSIAN NAIVE BAYES
// ----------------------------------------------------

export interface NaiveBayesResult {
  priors: { 0: number; 1: number };
  means: { [label: number]: { x: number; y: number } };
  variances: { [label: number]: { x: number; y: number } };
  accuracy: number;
  predictProb: (x: number, y: number) => { prob0: number; prob1: number };
}

export function fitGaussianNaiveBayes(dots: DataDot[], varSmoothing = 0.01): NaiveBayesResult {
  const n = dots.length;
  if (n === 0) {
    return {
      priors: { 0: 0.5, 1: 0.5 },
      means: { 0: { x: 0.5, y: 0.5 }, 1: { x: 0.5, y: 0.5 } },
      variances: { 0: { x: 0.1, y: 0.1 }, 1: { x: 0.1, y: 0.1 } },
      accuracy: 100,
      predictProb: () => ({ prob0: 0.5, prob1: 0.5 }),
    };
  }

  const class0 = dots.filter((d) => (d.label ?? 0) === 0);
  const class1 = dots.filter((d) => (d.label ?? 0) === 1);

  const prior0 = (class0.length || 1) / n;
  const prior1 = (class1.length || 1) / n;

  const mean0x = class0.length > 0 ? class0.reduce((s, d) => s + d.x, 0) / class0.length : 0.3;
  const mean0y = class0.length > 0 ? class0.reduce((s, d) => s + d.y, 0) / class0.length : 0.3;
  const mean1x = class1.length > 0 ? class1.reduce((s, d) => s + d.x, 0) / class1.length : 0.7;
  const mean1y = class1.length > 0 ? class1.reduce((s, d) => s + d.y, 0) / class1.length : 0.7;

  const var0x = (class0.reduce((s, d) => s + (d.x - mean0x) ** 2, 0) / Math.max(1, class0.length)) + varSmoothing;
  const var0y = (class0.reduce((s, d) => s + (d.y - mean0y) ** 2, 0) / Math.max(1, class0.length)) + varSmoothing;
  const var1x = (class1.reduce((s, d) => s + (d.x - mean1x) ** 2, 0) / Math.max(1, class1.length)) + varSmoothing;
  const var1y = (class1.reduce((s, d) => s + (d.y - mean1y) ** 2, 0) / Math.max(1, class1.length)) + varSmoothing;

  const gaussian = (v: number, m: number, varVal: number) => {
    const denom = Math.sqrt(2 * Math.PI * varVal);
    const num = Math.exp(-((v - m) ** 2) / (2 * varVal));
    return Math.max(1e-9, num / denom);
  };

  const predictProb = (x: number, y: number) => {
    const like0 = prior0 * gaussian(x, mean0x, var0x) * gaussian(y, mean0y, var0y);
    const like1 = prior1 * gaussian(x, mean1x, var1x) * gaussian(y, mean1y, var1y);
    const sum = like0 + like1 || 1e-9;
    return { prob0: like0 / sum, prob1: like1 / sum };
  };

  let correct = 0;
  dots.forEach((d) => {
    const { prob0, prob1 } = predictProb(d.x, d.y);
    const pred = prob1 >= prob0 ? 1 : 0;
    if (pred === (d.label ?? 0)) correct++;
  });

  return {
    priors: { 0: prior0, 1: prior1 },
    means: { 0: { x: mean0x, y: mean0y }, 1: { x: mean1x, y: mean1y } },
    variances: { 0: { x: var0x, y: var0y }, 1: { x: var1x, y: var1y } },
    accuracy: Math.round((correct / n) * 100),
    predictProb,
  };
}

// ----------------------------------------------------
// 11. RANDOM FOREST ENSEMBLE
// ----------------------------------------------------

export interface RandomForestResult {
  treesCount: number;
  accuracy: number;
  predictProb: (x: number, y: number) => number;
}

export function trainRandomForest(
  dots: DataDot[],
  numTrees = 5,
  maxDepth = 3
): RandomForestResult {
  const n = dots.length;
  if (n === 0) {
    return { treesCount: 0, accuracy: 100, predictProb: () => 0.5 };
  }

  // Train B bootstrap trees
  const treePartitionsList: TreePartition[][] = [];
  for (let b = 0; b < numTrees; b++) {
    const sample: DataDot[] = [];
    for (let i = 0; i < n; i++) {
      sample.push(dots[Math.floor(Math.random() * n)]);
    }
    const { partitions } = buildDecisionTreePartitions(sample, maxDepth, 'gini');
    treePartitionsList.push(partitions);
  }

  const predictProb = (x: number, y: number) => {
    let votes1 = 0;
    treePartitionsList.forEach((partitions) => {
      const match = partitions.find(
        (p: TreePartition) => x >= p.xMin && x <= p.xMax && y >= p.yMin && y <= p.yMax
      );
      if (match && match.predictedLabel === 1) votes1++;
    });
    return votes1 / numTrees;
  };

  let correct = 0;
  dots.forEach((d) => {
    const p1 = predictProb(d.x, d.y);
    const pred = p1 >= 0.5 ? 1 : 0;
    if (pred === (d.label ?? 0)) correct++;
  });

  return {
    treesCount: numTrees,
    accuracy: Math.round((correct / n) * 100),
    predictProb,
  };
}

// ----------------------------------------------------
// 12. HIERARCHICAL / AGGLOMERATIVE CLUSTERING
// ----------------------------------------------------

export interface HierarchicalResult {
  clusters: number[];
  numClusters: number;
  mergeDistance: number;
}

export function runHierarchicalClustering(
  dots: DataDot[],
  targetK = 3,
  linkage: 'single' | 'complete' | 'average' = 'average'
): HierarchicalResult {
  const n = dots.length;
  if (n === 0) return { clusters: [], numClusters: 0, mergeDistance: 0 };

  let currentClusters: number[][] = dots.map((_, i) => [i]);
  let lastDist = 0;

  while (currentClusters.length > Math.max(1, Math.min(targetK, n))) {
    let minDist = Infinity;
    let mergeI = 0;
    let mergeJ = 1;

    for (let i = 0; i < currentClusters.length; i++) {
      for (let j = i + 1; j < currentClusters.length; j++) {
        const cA = currentClusters[i];
        const cB = currentClusters[j];

        let d = 0;
        if (linkage === 'single') {
          d = Infinity;
          cA.forEach((idxA) => {
            cB.forEach((idxB) => {
              const dist = Math.hypot(dots[idxA].x - dots[idxB].x, dots[idxA].y - dots[idxB].y);
              if (dist < d) d = dist;
            });
          });
        } else if (linkage === 'complete') {
          d = -Infinity;
          cA.forEach((idxA) => {
            cB.forEach((idxB) => {
              const dist = Math.hypot(dots[idxA].x - dots[idxB].x, dots[idxA].y - dots[idxB].y);
              if (dist > d) d = dist;
            });
          });
        } else {
          let sum = 0;
          cA.forEach((idxA) => {
            cB.forEach((idxB) => {
              sum += Math.hypot(dots[idxA].x - dots[idxB].x, dots[idxA].y - dots[idxB].y);
            });
          });
          d = sum / (cA.length * cB.length);
        }

        if (d < minDist) {
          minDist = d;
          mergeI = i;
          mergeJ = j;
        }
      }
    }

    lastDist = minDist;
    currentClusters[mergeI] = [...currentClusters[mergeI], ...currentClusters[mergeJ]];
    currentClusters.splice(mergeJ, 1);
  }

  const clusterMapping: number[] = new Array(n).fill(0);
  currentClusters.forEach((cl, clId) => {
    cl.forEach((idx) => {
      clusterMapping[idx] = clId;
    });
  });

  return {
    clusters: clusterMapping,
    numClusters: currentClusters.length,
    mergeDistance: Number(lastDist.toFixed(4)),
  };
}
