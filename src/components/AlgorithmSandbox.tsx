import React, { useEffect, useRef, useState, useCallback } from 'react';
import { AlgorithmId, AlgorithmParams, Centroid, DataDot } from '../types/ml';
import { ALGORITHMS } from '../utils/algorithms';
import { generateDataset, DatasetPreset, DATASET_PRESETS } from '../utils/datasets';
import {
  initCentroids,
  runKMeansStep,
  computeOLS,
  stepGradientDescent,
  GradientDescentState,
  queryKNN,
  solveSVM,
  computePCA,
  buildDecisionTreePartitions,
  initNeuralNet,
  trainEpochNN,
  forwardPassNN,
  NeuralNetWeights,
  runDBSCAN,
  solveLogisticRegression,
  fitGaussianNaiveBayes,
  trainRandomForest,
  runHierarchicalClustering,
} from '../utils/math';
import { DotInspector } from './DotInspector';
import { MetricsDashboard, PerformanceMetrics } from './MetricsDashboard';
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Plus,
  Move,
  Radar,
  Sliders,
  CheckCircle,
  HelpCircle,
  Bot,
  Gauge,
  FastForward,
  ChevronDown,
  Database,
  UploadCloud,
} from 'lucide-react';

interface AlgorithmSandboxProps {
  algorithmId: AlgorithmId;
  onSelectAlgorithm: (id: AlgorithmId) => void;
  onOpenChat?: () => void;
  customDots?: DataDot[];
  customDatasetName?: string;
  onOpenIngestModal?: () => void;
  onClearCustomDataset?: () => void;
}

export const AlgorithmSandbox: React.FC<AlgorithmSandboxProps> = ({
  algorithmId,
  onSelectAlgorithm,
  onOpenChat,
  customDots,
  customDatasetName,
  onOpenIngestModal,
  onClearCustomDataset,
}) => {
  const currentAlgo = ALGORITHMS.find((a) => a.id === algorithmId) || ALGORITHMS[0];

  // Data dots state
  const [dots, setDots] = useState<DataDot[]>(() => customDots || generateDataset('blobs', 50, 0.04));
  const [datasetPreset, setDatasetPreset] = useState<DatasetPreset>('blobs');
  const activePresetInfo = DATASET_PRESETS.find((p) => p.id === datasetPreset);
  const [selectedDot, setSelectedDot] = useState<DataDot | null>(null);
  const [activeTool, setActiveTool] = useState<'move' | 'add0' | 'add1' | 'probe'>('move');

  // Sync customDots if changed externally
  useEffect(() => {
    if (customDots && customDots.length > 0) {
      setDots(customDots);
    }
  }, [customDots]);

  // Simulation controls
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [simSpeed, setSimSpeed] = useState<number>(1);
  const [stepCount, setStepCount] = useState<number>(0);

  // Parameter state
  const [params, setParams] = useState<AlgorithmParams>({
    kmeansK: 3,
    kmeansInit: 'kmeans++',
    showVoronoi: true,
    knnK: 5,
    knnWeighting: 'uniform',
    showDecisionBoundary: true,
    learningRate: 0.1,
    regIterations: 0,
    fitIntercept: true,
    showResiduals: true,
    svmC: 1.0,
    svmKernel: 'linear',
    svmRbfGamma: 5.0,
    nnHiddenNeurons: 4,
    nnActivation: 'relu',
    nnEpochs: 0,
    pcaComponents: 1,
    showProjections: true,
    treeMaxDepth: 3,
    treeCriterion: 'gini',
    dbscanEps: 0.14,
    dbscanMinPts: 4,
    logRegLr: 0.15,
    logRegThreshold: 0.5,
    nbVarSmoothing: 0.01,
    rfNumTrees: 5,
    rfMaxDepth: 3,
    hierarchicalClusters: 3,
    hierarchicalLinkage: 'average',
  });

  // Algorithm-specific state refs and storage
  const [centroids, setCentroids] = useState<Centroid[]>([]);
  const [gdState, setGdState] = useState<GradientDescentState>({
    theta0: 0.2,
    theta1: 0.1,
    loss: 0.5,
    history: [],
  });
  const [nnWeights, setNnWeights] = useState<NeuralNetWeights>(() => initNeuralNet(4));
  const [nnLoss, setNnLoss] = useState<number>(0.5);

  // Probe cursor coordinates (0-1) for KNN
  const [probePos, setProbePos] = useState<{ x: number; y: number }>({ x: 0.5, y: 0.5 });

  // Performance telemetry history for sparklines & counters
  const [metricHistory, setMetricHistory] = useState<number[]>([]);

  // Dragging dot state
  const draggingDotIdRef = useRef<string | null>(null);

  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const animLoopRef = useRef<number | null>(null);

  // Re-seed data on preset change or algorithm recommendation
  const handleSetPreset = (preset: DatasetPreset) => {
    setDatasetPreset(preset);
    let count = 50;
    if (preset === 'linear') count = 40;
    const newDots = generateDataset(preset, count, 0.05);
    setDots(newDots);
    setSelectedDot(null);
    setStepCount(0);

    // Reset centroids if in kmeans
    if (algorithmId === 'kmeans') {
      setCentroids(initCentroids(newDots, params.kmeansK, params.kmeansInit));
    }
  };

  // Switch algorithm recommendations
  useEffect(() => {
    // When switching algorithms, choose a sensible default preset
    let defaultPreset: DatasetPreset = 'blobs';
    if (algorithmId === 'linear-regression') defaultPreset = 'linear';
    if (algorithmId === 'svm' || algorithmId === 'neural-net' || algorithmId === 'dbscan') defaultPreset = 'moons';
    if (algorithmId === 'random-forest') defaultPreset = 'xor';
    if (algorithmId === 'pca') defaultPreset = 'blobs';
    handleSetPreset(defaultPreset);
    setIsPlaying(false);
  }, [algorithmId]);

  // Sync K-Means centroids when K or mode changes
  useEffect(() => {
    if (algorithmId === 'kmeans') {
      setCentroids(initCentroids(dots, params.kmeansK, params.kmeansInit));
      setStepCount(0);
    }
  }, [params.kmeansK, params.kmeansInit, algorithmId]);

  // Sync Neural Net weights when neuron count changes
  useEffect(() => {
    if (algorithmId === 'neural-net') {
      setNnWeights(initNeuralNet(params.nnHiddenNeurons));
      setStepCount(0);
    }
  }, [params.nnHiddenNeurons, algorithmId]);

  // Execute 1 Step
  const stepSimulation = useCallback(() => {
    setStepCount((s) => s + 1);

    if (algorithmId === 'kmeans') {
      const res = runKMeansStep(dots, centroids);
      setCentroids(res.centroids);
      setDots(res.dots);
      if (res.converged) setIsPlaying(false);
    } else if (algorithmId === 'linear-regression') {
      setGdState((prev) => stepGradientDescent(dots, prev, params.learningRate));
    } else if (algorithmId === 'neural-net') {
      const res = trainEpochNN(dots, nnWeights, params.learningRate, params.nnActivation);
      setNnWeights(res.updatedWeights);
      setNnLoss(res.loss);
    }
  }, [algorithmId, dots, centroids, params, nnWeights]);

  // Playback loop
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      stepSimulation();
    }, Math.max(50, 400 / simSpeed));

    return () => clearInterval(interval);
  }, [isPlaying, simSpeed, stepSimulation]);

  // --------------------------------------------------------------------------
  // Canvas Rendering Loop
  // --------------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Helper coordinate conversion
    const toCanvasX = (normX: number) => normX * width;
    const toCanvasY = (normY: number) => (1 - normY) * height; // Invert Y for cartesian
    const fromCanvasX = (cx: number) => Math.max(0, Math.min(1, cx / width));
    const fromCanvasY = (cy: number) => Math.max(0, Math.min(1, 1 - cy / height));

    ctx.clearRect(0, 0, width, height);

    // 1. Draw subtle background grid
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.35)';
    ctx.lineWidth = 1;
    const gridStep = 40;
    for (let x = 0; x < width; x += gridStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // 2. Algorithm-specific background fields & decision surfaces
    if (algorithmId === 'kmeans' && params.showVoronoi && centroids.length > 0) {
      // Draw Voronoi Cell partitions
      const cellW = 12;
      const cellH = 12;
      for (let x = 0; x < width; x += cellW) {
        for (let y = 0; y < height; y += cellH) {
          const nx = fromCanvasX(x + cellW / 2);
          const ny = fromCanvasY(y + cellH / 2);
          let minDistSq = Infinity;
          let bestCentroid = centroids[0];
          centroids.forEach((c) => {
            const d = (nx - c.x) ** 2 + (ny - c.y) ** 2;
            if (d < minDistSq) {
              minDistSq = d;
              bestCentroid = c;
            }
          });
          ctx.fillStyle = bestCentroid.color + '12';
          ctx.fillRect(x, y, cellW, cellH);
        }
      }
    } else if (algorithmId === 'knn' && params.showDecisionBoundary) {
      // Draw KNN decision contour field
      const step = 16;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = fromCanvasX(x + step / 2);
          const ny = fromCanvasY(y + step / 2);
          const q = queryKNN(dots, nx, ny, params.knnK, params.knnWeighting);
          const prob1 = q.probabilities[1] ?? 0.5;
          ctx.fillStyle =
            prob1 > 0.5
              ? `rgba(245, 158, 11, ${(prob1 - 0.5) * 0.35})`
              : `rgba(6, 182, 212, ${(0.5 - prob1) * 0.35})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algorithmId === 'svm') {
      const svmRes = solveSVM(dots, params.svmC, params.svmKernel);
      // Draw separating hyperplane w1*x + w2*y + b = 0
      const [w1, w2] = svmRes.weights;
      const b = svmRes.bias;

      if (Math.abs(w2) > 0.001) {
        // y = -(w1*x + b) / w2
        const getY = (nx: number, offset: number) => -(w1 * nx + b + offset) / w2;

        // Hyperplane
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(getY(0, 0)));
        ctx.lineTo(toCanvasX(1), toCanvasY(getY(1, 0)));
        ctx.stroke();

        // Upper Margin (offset = +1)
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(getY(0, 1)));
        ctx.lineTo(toCanvasX(1), toCanvasY(getY(1, 1)));
        ctx.stroke();

        // Lower Margin (offset = -1)
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(getY(0, -1)));
        ctx.lineTo(toCanvasX(1), toCanvasY(getY(1, -1)));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else if (algorithmId === 'decision-tree') {
      const { partitions, splits } = buildDecisionTreePartitions(
        dots,
        params.treeMaxDepth,
        params.treeCriterion
      );

      // Fill partition regions with subtle class colors
      partitions.forEach((part) => {
        const px = toCanvasX(part.xMin);
        const py = toCanvasY(part.yMax);
        const pw = toCanvasX(part.xMax) - px;
        const ph = toCanvasY(part.yMin) - py;

        const color = part.predictedLabel === 1 ? '245, 158, 11' : '6, 182, 212';
        ctx.fillStyle = `rgba(${color}, ${0.08 + part.purity * 0.12})`;
        ctx.fillRect(px, py, pw, ph);

        // Partition border
        ctx.strokeStyle = `rgba(${color}, 0.25)`;
        ctx.lineWidth = 1;
        ctx.strokeRect(px, py, pw, ph);
      });

      // Split cut lines
      splits.forEach((split) => {
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        if (split.feature === 'x') {
          const sx = toCanvasX(split.threshold);
          ctx.moveTo(sx, 0);
          ctx.lineTo(sx, height);
        } else {
          const sy = toCanvasY(split.threshold);
          ctx.moveTo(0, sy);
          ctx.lineTo(width, sy);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      });
    } else if (algorithmId === 'neural-net') {
      // Neural Net Decision Heatmap
      const step = 18;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = fromCanvasX(x + step / 2);
          const ny = fromCanvasY(y + step / 2);
          const { output } = forwardPassNN(nx, ny, nnWeights, params.nnActivation);
          ctx.fillStyle =
            output > 0.5
              ? `rgba(236, 72, 153, ${(output - 0.5) * 0.4})`
              : `rgba(6, 182, 212, ${(0.5 - output) * 0.4})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algorithmId === 'linear-regression') {
      // Draw regression line & residual sticks
      const ols = computeOLS(dots);
      const m = isPlaying || stepCount > 0 ? gdState.theta1 : ols.slope;
      const c = isPlaying || stepCount > 0 ? gdState.theta0 : ols.intercept;

      // Residual error sticks connecting dots to line
      if (params.showResiduals) {
        dots.forEach((d) => {
          const yHat = m * d.x + c;
          const cx = toCanvasX(d.x);
          const cy = toCanvasY(d.y);
          const lineY = toCanvasY(yHat);

          ctx.strokeStyle = 'rgba(244, 63, 94, 0.45)';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx, lineY);
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }

      // Regression line
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(c));
      ctx.lineTo(toCanvasX(1), toCanvasY(m * 1 + c));
      ctx.stroke();
    } else if (algorithmId === 'pca') {
      const pca = computePCA(dots);
      const mx = toCanvasX(pca.mean.x);
      const my = toCanvasY(pca.mean.y);

      // Draw Principal Component Vector 1 (Green)
      const len1 = 200 * Math.sqrt(pca.varianceExplained1);
      const pc1EndX = toCanvasX(pca.mean.x + pca.pc1.x * 0.4);
      const pc1EndY = toCanvasY(pca.mean.y + pca.pc1.y * 0.4);
      const pc1StartX = toCanvasX(pca.mean.x - pca.pc1.x * 0.4);
      const pc1StartY = toCanvasY(pca.mean.y - pca.pc1.y * 0.4);

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pc1StartX, pc1StartY);
      ctx.lineTo(pc1EndX, pc1EndY);
      ctx.stroke();

      // Draw Principal Component Vector 2 (Orthogonal)
      const pc2EndX = toCanvasX(pca.mean.x + pca.pc2.x * 0.2);
      const pc2EndY = toCanvasY(pca.mean.y + pca.pc2.y * 0.2);
      const pc2StartX = toCanvasX(pca.mean.x - pca.pc2.x * 0.2);
      const pc2StartY = toCanvasY(pca.mean.y - pca.pc2.y * 0.2);

      ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pc2StartX, pc2StartY);
      ctx.lineTo(pc2EndX, pc2EndY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Center of mass mean star
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(mx, my, 5, 0, Math.PI * 2);
      ctx.fill();

      // Orthogonal projection shadows onto PC1
      if (params.showProjections) {
        pca.projectedDots.forEach((d) => {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(toCanvasX(d.x), toCanvasY(d.y));
          ctx.lineTo(toCanvasX(d.projX), toCanvasY(d.projY));
          ctx.stroke();
          ctx.setLineDash([]);

          // Ghost dot on projection line
          ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
          ctx.beginPath();
          ctx.arc(toCanvasX(d.projX), toCanvasY(d.projY), 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    } else if (algorithmId === 'dbscan') {
      const dbRes = runDBSCAN(dots, params.dbscanEps, params.dbscanMinPts);
      const epsPx = params.dbscanEps * width;

      // Draw epsilon neighborhood circles for Core dots
      dbRes.dots.forEach((d) => {
        if (d.dbscanType === 'core') {
          const cx = toCanvasX(d.x);
          const cy = toCanvasY(d.y);
          const col = d.cluster !== undefined ? dbRes.clusterColors[d.cluster % dbRes.clusterColors.length] : '#14b8a6';

          ctx.strokeStyle = col + '25';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(cx, cy, epsPx, 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = col + '08';
          ctx.fill();
        }
      });
    } else if (algorithmId === 'logistic-regression') {
      const logRes = solveLogisticRegression(dots, params.logRegLr, 60, params.logRegThreshold);
      const step = 16;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = fromCanvasX(x + step / 2);
          const ny = fromCanvasY(y + step / 2);
          const prob = logRes.predictProb(nx, ny);
          ctx.fillStyle =
            prob > 0.5
              ? `rgba(99, 102, 241, ${(prob - 0.5) * 0.45})`
              : `rgba(6, 182, 212, ${(0.5 - prob) * 0.45})`;
          ctx.fillRect(x, y, step, step);
        }
      }

      // 0.5 Decision Boundary line: w1*x + w2*y + b = 0
      if (Math.abs(logRes.w2) > 0.001) {
        const getY = (nx: number) => -(logRes.w1 * nx + logRes.b) / logRes.w2;
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(0), toCanvasY(getY(0)));
        ctx.lineTo(toCanvasX(1), toCanvasY(getY(1)));
        ctx.stroke();
      }
    } else if (algorithmId === 'naive-bayes') {
      const nbRes = fitGaussianNaiveBayes(dots, params.nbVarSmoothing);
      // Heatmap of posterior probabilities
      const step = 18;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = fromCanvasX(x + step / 2);
          const ny = fromCanvasY(y + step / 2);
          const { prob1 } = nbRes.predictProb(nx, ny);
          ctx.fillStyle =
            prob1 > 0.5
              ? `rgba(234, 179, 8, ${(prob1 - 0.5) * 0.35})`
              : `rgba(6, 182, 212, ${(0.5 - prob1) * 0.35})`;
          ctx.fillRect(x, y, step, step);
        }
      }

      // Elliptical Gaussian contours around Class 0 and Class 1 means
      [0, 1].forEach((cls) => {
        const m = nbRes.means[cls];
        const v = nbRes.variances[cls];
        const mx = toCanvasX(m.x);
        const my = toCanvasY(m.y);
        const rx = Math.sqrt(v.x) * width;
        const ry = Math.sqrt(v.y) * height;

        ctx.strokeStyle = cls === 1 ? 'rgba(234, 179, 8, 0.6)' : 'rgba(6, 182, 212, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(mx, my, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = cls === 1 ? '#eab308' : '#06b6d4';
        ctx.beginPath();
        ctx.arc(mx, my, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    } else if (algorithmId === 'random-forest') {
      const rfRes = trainRandomForest(dots, params.rfNumTrees, params.rfMaxDepth);
      const step = 16;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = fromCanvasX(x + step / 2);
          const ny = fromCanvasY(y + step / 2);
          const prob1 = rfRes.predictProb(nx, ny);
          ctx.fillStyle =
            prob1 > 0.5
              ? `rgba(16, 185, 129, ${(prob1 - 0.5) * 0.4})`
              : `rgba(6, 182, 212, ${(0.5 - prob1) * 0.4})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algorithmId === 'hierarchical') {
      const hRes = runHierarchicalClustering(dots, params.hierarchicalClusters, params.hierarchicalLinkage);
      const colors = ['#f43f5e', '#06b6d4', '#f59e0b', '#8b5cf6', '#10b981'];

      // Draw cluster connection lines between nearest points in same cluster
      for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
          if (hRes.clusters[i] === hRes.clusters[j]) {
            const dist = Math.hypot(dots[i].x - dots[j].x, dots[i].y - dots[j].y);
            if (dist < 0.2) {
              const cl = hRes.clusters[i];
              ctx.strokeStyle = colors[cl % colors.length] + '35';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(toCanvasX(dots[i].x), toCanvasY(dots[i].y));
              ctx.lineTo(toCanvasX(dots[j].x), toCanvasY(dots[j].y));
              ctx.stroke();
            }
          }
        }
      }
    }

    // 3. Render KNN Probe Cursor & Radial Search Lines
    if (algorithmId === 'knn' && activeTool === 'probe') {
      const qRes = queryKNN(dots, probePos.x, probePos.y, params.knnK, params.knnWeighting);
      const px = toCanvasX(probePos.x);
      const py = toCanvasY(probePos.y);

      // Distance lines to K nearest neighbors
      qRes.neighbors.forEach((n) => {
        const nx = toCanvasX(n.dot.x);
        const ny = toCanvasY(n.dot.y);
        ctx.strokeStyle = n.dot.label === 1 ? 'rgba(245, 158, 11, 0.7)' : 'rgba(6, 182, 212, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(nx, ny);
        ctx.stroke();
      });

      // Probe ring
      const maxDist = qRes.neighbors.length > 0 ? qRes.neighbors[qRes.neighbors.length - 1].distance : 0.2;
      const radiusPx = maxDist * width;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(px, py, radiusPx, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Probe center cursor
      ctx.fillStyle = qRes.predictedLabel === 1 ? '#f59e0b' : '#06b6d4';
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // 4. Render Centroids (in K-Means)
    if (algorithmId === 'kmeans') {
      centroids.forEach((c) => {
        const cx = toCanvasX(c.x);
        const cy = toCanvasY(c.y);

        // Centroid outer halo
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.stroke();

        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(cx - 16, cy);
        ctx.lineTo(cx + 16, cy);
        ctx.moveTo(cx, cy - 16);
        ctx.lineTo(cx, cy + 16);
        ctx.stroke();

        // Inner solid
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // 5. Render All Data Dots
    const dbscanResult = algorithmId === 'dbscan' ? runDBSCAN(dots, params.dbscanEps, params.dbscanMinPts) : null;
    const hierarchicalResult =
      algorithmId === 'hierarchical'
        ? runHierarchicalClustering(dots, params.hierarchicalClusters, params.hierarchicalLinkage)
        : null;
    const hierPalette = ['#f43f5e', '#06b6d4', '#f59e0b', '#8b5cf6', '#10b981'];

    dots.forEach((d, idx) => {
      const cx = toCanvasX(d.x);
      const cy = toCanvasY(d.y);
      const isSelected = selectedDot?.id === d.id;

      let dotColor = '#06b6d4'; // default cyan
      if (algorithmId === 'kmeans' && d.cluster !== undefined && centroids[d.cluster]) {
        dotColor = centroids[d.cluster].color;
      } else if (algorithmId === 'dbscan' && dbscanResult) {
        const dbDot = dbscanResult.dots[idx];
        if (dbDot?.dbscanType === 'noise') {
          dotColor = '#64748b'; // slate-500 for noise
        } else if (dbDot?.cluster !== undefined) {
          dotColor = dbscanResult.clusterColors[dbDot.cluster % dbscanResult.clusterColors.length];
        }
      } else if (algorithmId === 'hierarchical' && hierarchicalResult) {
        const clId = hierarchicalResult.clusters[idx] ?? 0;
        dotColor = hierPalette[clId % hierPalette.length];
      } else if (d.label === 1) {
        dotColor = '#f59e0b'; // amber
      } else if (algorithmId === 'linear-regression') {
        dotColor = '#38bdf8';
      }

      // Support vector glow highlight
      if (algorithmId === 'svm' && d.isSupportVector) {
        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Selected ring
      if (isSelected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 11, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Main dot body
      ctx.fillStyle = dotColor;
      ctx.shadowColor = dotColor;
      ctx.shadowBlur = isSelected ? 12 : 5;
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? 6.5 : 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }, [
    dots,
    centroids,
    algorithmId,
    params,
    selectedDot,
    activeTool,
    probePos,
    gdState,
    nnWeights,
    isPlaying,
    stepCount,
  ]);

  // --------------------------------------------------------------------------
  // Loss Surface Canvas (for Linear Regression Gradient Descent)
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (algorithmId !== 'linear-regression') return;
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw Parabolic Bowl Contours (Loss Surface)
    const centerX = w / 2;
    const centerY = h / 2;

    [15, 30, 48, 68, 90].forEach((r, idx) => {
      ctx.strokeStyle = `rgba(59, 130, 246, ${0.15 + idx * 0.08})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, r * 1.3, r, 0.2, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Optimum Global Minimum Target
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
    ctx.fill();

    // Parameter Trajectory Path down the bowl
    if (gdState.history.length > 1) {
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      gdState.history.forEach((pt, i) => {
        const px = centerX + (pt.theta1 - 0.7) * 90;
        const py = centerY + (pt.theta0 - 0.15) * 90;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // Current Parameter Dot
    const currentPx = centerX + (gdState.theta1 - 0.7) * 90;
    const currentPy = centerY + (gdState.theta0 - 0.15) * 90;
    ctx.fillStyle = '#f43f5e';
    ctx.shadowColor = '#f43f5e';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(currentPx, currentPy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }, [algorithmId, gdState]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current || !containerRef.current) return;
      canvasRef.current.width = containerRef.current.clientWidth;
      canvasRef.current.height = containerRef.current.clientHeight;
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --------------------------------------------------------------------------
  // Canvas Mouse & Interaction Handlers
  // --------------------------------------------------------------------------
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { nx: 0, ny: 0, cx: 0, cy: 0 };
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const nx = Math.max(0, Math.min(1, cx / canvas.width));
    const ny = Math.max(0, Math.min(1, 1 - cy / canvas.height));
    return { nx, ny, cx, cy };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { nx, ny } = getCanvasCoords(e);

    if (activeTool === 'probe') {
      setProbePos({ x: nx, y: ny });
      return;
    }

    if (activeTool === 'add0' || activeTool === 'add1') {
      const newDot: DataDot = {
        id: `dot-${Date.now()}`,
        x: nx,
        y: ny,
        label: activeTool === 'add0' ? 0 : 1,
        continuousTarget: ny,
      };
      setDots((prev) => [...prev, newDot]);
      setSelectedDot(newDot);
      return;
    }

    // Check if clicked an existing dot
    let clickedDot: DataDot | null = null;
    let minD = 0.04; // pick radius in normalized coords

    dots.forEach((d) => {
      const dist = Math.hypot(d.x - nx, d.y - ny);
      if (dist < minD) {
        minD = dist;
        clickedDot = d;
      }
    });

    if (clickedDot) {
      setSelectedDot(clickedDot);
      draggingDotIdRef.current = (clickedDot as DataDot).id;
    } else {
      setSelectedDot(null);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { nx, ny } = getCanvasCoords(e);

    if (activeTool === 'probe') {
      setProbePos({ x: nx, y: ny });
      return;
    }

    if (draggingDotIdRef.current) {
      setDots((prev) =>
        prev.map((d) => {
          if (d.id === draggingDotIdRef.current) {
            return { ...d, x: nx, y: ny, continuousTarget: ny };
          }
          return d;
        })
      );
      if (selectedDot && selectedDot.id === draggingDotIdRef.current) {
        setSelectedDot((s) => (s ? { ...s, x: nx, y: ny } : null));
      }
    }
  };

  const handleMouseUp = () => {
    draggingDotIdRef.current = null;
  };

  // Metrics calculation for the real-time telemetry dashboard
  const getMetrics = (): PerformanceMetrics => {
    switch (algorithmId) {
      case 'kmeans': {
        const inert = dots.reduce((sum, d) => {
          if (d.cluster !== undefined && centroids[d.cluster]) {
            const c = centroids[d.cluster];
            return sum + (d.x - c.x) ** 2 + (d.y - c.y) ** 2;
          }
          return sum;
        }, 0);
        return {
          primaryLabel: 'Inertia (WCSS)',
          primaryValue: inert,
          primaryDecimals: 4,
          secondaryLabel: 'Clusters (K)',
          secondaryValue: params.kmeansK,
          secondaryDecimals: 0,
          tertiaryLabel: 'Step Count',
          tertiaryValue: stepCount,
          tertiaryDecimals: 0,
          healthStatus: isPlaying ? 'training' : inert < 0.2 ? 'optimal' : 'idle',
          statusText: isPlaying ? 'Centroids Migrating' : 'Converged State',
          history: metricHistory,
        };
      }

      case 'linear-regression': {
        const ols = computeOLS(dots);
        const currentLoss = isPlaying || stepCount > 0 ? gdState.loss : ols.mse;
        return {
          primaryLabel: 'Loss (MSE)',
          primaryValue: currentLoss,
          primaryDecimals: 4,
          secondaryLabel: 'R² Fit Score',
          secondaryValue: ols.r2 * 100,
          secondarySuffix: '%',
          secondaryDecimals: 1,
          tertiaryLabel: 'Iteration Step',
          tertiaryValue: stepCount,
          tertiaryDecimals: 0,
          healthStatus: currentLoss < 0.05 ? 'optimal' : isPlaying ? 'training' : 'warning',
          statusText:
            currentLoss < 0.05
              ? 'Optimal Fit'
              : isPlaying
              ? 'Descending Gradient'
              : 'Residual Strain',
          history: metricHistory,
        };
      }

      case 'knn': {
        const qRes = queryKNN(dots, probePos.x, probePos.y, params.knnK, params.knnWeighting);
        const conf = Math.max(qRes.probabilities[0], qRes.probabilities[1]) * 100;
        return {
          primaryLabel: 'Confidence',
          primaryValue: conf,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'Margin',
          secondaryValue: qRes.decisionMargin,
          secondaryDecimals: 3,
          tertiaryLabel: 'Neighbors (K)',
          tertiaryValue: params.knnK,
          tertiaryDecimals: 0,
          healthStatus: conf > 75 ? 'optimal' : 'warning',
          statusText: conf > 75 ? 'Clear Consensus' : 'Ambiguous Boundary',
          history: metricHistory,
        };
      }

      case 'svm': {
        const svmRes = solveSVM(dots, params.svmC, params.svmKernel);
        return {
          primaryLabel: 'Accuracy',
          primaryValue: svmRes.accuracy,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'Margin Width',
          secondaryValue: svmRes.marginWidth,
          secondaryDecimals: 3,
          tertiaryLabel: 'Support Vectors',
          tertiaryValue: svmRes.supportVectors.length,
          tertiaryDecimals: 0,
          healthStatus: svmRes.accuracy > 85 ? 'optimal' : 'warning',
          statusText: svmRes.accuracy > 85 ? 'Hyperplane Separated' : 'Soft Margin Overlap',
          history: metricHistory,
        };
      }

      case 'pca': {
        const pca = computePCA(dots);
        return {
          primaryLabel: 'PC1 Variance',
          primaryValue: pca.varianceExplained1 * 100,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'PC2 Variance',
          secondaryValue: pca.varianceExplained2 * 100,
          secondarySuffix: '%',
          secondaryDecimals: 1,
          tertiaryLabel: 'Dimensions',
          tertiaryValue: params.pcaComponents,
          tertiaryDecimals: 0,
          healthStatus: 'optimal',
          statusText: `${(pca.varianceExplained1 * 100).toFixed(0)}% Variance Retained`,
          history: metricHistory,
        };
      }

      case 'decision-tree': {
        const dt = buildDecisionTreePartitions(dots, params.treeMaxDepth, params.treeCriterion);
        const avgPurity =
          dt.partitions.reduce((acc, p) => acc + p.purity * p.count, 0) /
          Math.max(1, dots.length);
        return {
          primaryLabel: 'Tree Purity',
          primaryValue: avgPurity * 100,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'Partitions',
          secondaryValue: dt.partitions.length,
          secondaryDecimals: 0,
          tertiaryLabel: 'Max Depth',
          tertiaryValue: params.treeMaxDepth,
          tertiaryDecimals: 0,
          healthStatus: avgPurity > 0.85 ? 'optimal' : 'training',
          statusText:
            params.treeMaxDepth > 3 ? 'Deep Cuts (Overfit Risk)' : 'Balanced Partitioning',
          history: metricHistory,
        };
      }

      case 'neural-net': {
        return {
          primaryLabel: 'Loss (MSE)',
          primaryValue: nnLoss,
          primaryDecimals: 4,
          secondaryLabel: 'Hidden Units',
          secondaryValue: params.nnHiddenNeurons,
          secondaryDecimals: 0,
          tertiaryLabel: 'Epoch',
          tertiaryValue: stepCount,
          tertiaryDecimals: 0,
          healthStatus: nnLoss < 0.08 ? 'optimal' : isPlaying ? 'training' : 'warning',
          statusText:
            nnLoss < 0.08
              ? 'Low Loss Equilibrium'
              : isPlaying
              ? 'Backpropagating Gradients'
              : 'Untrained State',
          history: metricHistory,
        };
      }

      case 'dbscan': {
        const dbRes = runDBSCAN(dots, params.dbscanEps, params.dbscanMinPts);
        const noiseRatio = dots.length > 0 ? (dbRes.noiseCount / dots.length) * 100 : 0;
        return {
          primaryLabel: 'Clusters Found',
          primaryValue: dbRes.numClusters,
          primaryDecimals: 0,
          secondaryLabel: 'Noise Outliers',
          secondaryValue: noiseRatio,
          secondarySuffix: '%',
          secondaryDecimals: 1,
          tertiaryLabel: 'Core Dots',
          tertiaryValue: dbRes.coreCount,
          tertiaryDecimals: 0,
          healthStatus: noiseRatio < 25 ? 'optimal' : noiseRatio < 45 ? 'training' : 'warning',
          statusText: `${dbRes.numClusters} Dense Clusters (${dbRes.noiseCount} noise dots)`,
          history: metricHistory,
        };
      }

      case 'logistic-regression': {
        const logRes = solveLogisticRegression(dots, params.logRegLr, 60, params.logRegThreshold);
        return {
          primaryLabel: 'Log-Loss',
          primaryValue: logRes.logLoss,
          primaryDecimals: 4,
          secondaryLabel: 'Accuracy',
          secondaryValue: logRes.accuracy,
          secondarySuffix: '%',
          secondaryDecimals: 1,
          tertiaryLabel: 'Threshold',
          tertiaryValue: params.logRegThreshold,
          tertiaryDecimals: 2,
          healthStatus: logRes.logLoss < 0.35 ? 'optimal' : 'warning',
          statusText: `${logRes.accuracy}% Accuracy (BCE: ${logRes.logLoss.toFixed(3)})`,
          history: metricHistory,
        };
      }

      case 'naive-bayes': {
        const nbRes = fitGaussianNaiveBayes(dots, params.nbVarSmoothing);
        return {
          primaryLabel: 'Bayes Accuracy',
          primaryValue: nbRes.accuracy,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'Prior P(C₁)',
          secondaryValue: nbRes.priors[1] * 100,
          secondarySuffix: '%',
          secondaryDecimals: 1,
          tertiaryLabel: 'Smoothing (σ²)',
          tertiaryValue: params.nbVarSmoothing,
          tertiaryDecimals: 3,
          healthStatus: nbRes.accuracy > 80 ? 'optimal' : 'training',
          statusText: `${nbRes.accuracy}% Bayesian Accuracy`,
          history: metricHistory,
        };
      }

      case 'random-forest': {
        const rfRes = trainRandomForest(dots, params.rfNumTrees, params.rfMaxDepth);
        return {
          primaryLabel: 'Ensemble Accuracy',
          primaryValue: rfRes.accuracy,
          primarySuffix: '%',
          primaryDecimals: 1,
          secondaryLabel: 'Trees in Forest',
          secondaryValue: params.rfNumTrees,
          secondaryDecimals: 0,
          tertiaryLabel: 'Tree Depth',
          tertiaryValue: params.rfMaxDepth,
          tertiaryDecimals: 0,
          healthStatus: rfRes.accuracy > 85 ? 'optimal' : 'training',
          statusText: `${params.rfNumTrees} Bagged Trees Voting`,
          history: metricHistory,
        };
      }

      case 'hierarchical': {
        const hRes = runHierarchicalClustering(dots, params.hierarchicalClusters, params.hierarchicalLinkage);
        return {
          primaryLabel: 'Active Clusters',
          primaryValue: hRes.numClusters,
          primaryDecimals: 0,
          secondaryLabel: 'Merge Distance',
          secondaryValue: hRes.mergeDistance,
          secondaryDecimals: 4,
          tertiaryLabel: 'Linkage',
          tertiaryValue: params.hierarchicalClusters,
          tertiaryDecimals: 0,
          healthStatus: 'optimal',
          statusText: `Agglomerated into ${hRes.numClusters} Groups`,
          history: metricHistory,
        };
      }
    }
  };

  const currentMetrics = getMetrics();

  // Keep sparkline history updated on metrics changes
  useEffect(() => {
    setMetricHistory((prev) => {
      const next = [...prev, currentMetrics.primaryValue];
      return next.slice(-25);
    });
  }, [
    dots.length,
    stepCount,
    params.kmeansK,
    params.knnK,
    params.learningRate,
    params.svmC,
    params.treeMaxDepth,
    params.nnHiddenNeurons,
    gdState.loss,
    nnLoss,
  ]);

  return (
    <div className="flex flex-col h-[calc(100vh-61px)] bg-slate-950 text-slate-100">
      {/* Algorithm Navigation Switcher Bar */}
      <div className="flex items-center gap-2 px-6 py-2.5 bg-slate-900/80 border-b border-slate-800 overflow-x-auto shrink-0">
        <span className="text-xs font-medium text-slate-400 mr-2 whitespace-nowrap">
          Algorithm:
        </span>
        {ALGORITHMS.map((algo) => (
          <button
            key={algo.id}
            onClick={() => onSelectAlgorithm(algo.id)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap shrink-0 ${
              algorithmId === algo.id
                ? 'bg-slate-800 text-white shadow-sm ring-1 ring-slate-700'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: algo.color }}
            />
            <span>{algo.name}</span>
          </button>
        ))}
      </div>

      {/* Main 2-Zone Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Interactive Canvas Stage */}
        <div ref={containerRef} className="flex-1 relative bg-slate-950 flex flex-col">
          {/* Top Stage Toolbar */}
          <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur z-10">
            {/* Interactive Tool Selector */}
            <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setActiveTool('move')}
                className={`px-2.5 py-1 text-xs font-medium rounded flex items-center gap-1.5 transition-colors ${
                  activeTool === 'move' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Select and drag dots"
              >
                <Move className="w-3.5 h-3.5" />
                <span>Drag Dot</span>
              </button>

              <button
                onClick={() => setActiveTool('add0')}
                className={`px-2.5 py-1 text-xs font-medium rounded flex items-center gap-1.5 transition-colors ${
                  activeTool === 'add0'
                    ? 'bg-cyan-950/60 text-cyan-300 border border-cyan-800/50'
                    : 'text-slate-400 hover:text-cyan-300'
                }`}
                title="Add Class 0 / Target Dot"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>+ Cyan Dot</span>
              </button>

              <button
                onClick={() => setActiveTool('add1')}
                className={`px-2.5 py-1 text-xs font-medium rounded flex items-center gap-1.5 transition-colors ${
                  activeTool === 'add1'
                    ? 'bg-amber-950/60 text-amber-300 border border-amber-800/50'
                    : 'text-slate-400 hover:text-amber-300'
                }`}
                title="Add Class 1 Dot"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>+ Amber Dot</span>
              </button>

              {algorithmId === 'knn' && (
                <button
                  onClick={() => setActiveTool('probe')}
                  className={`px-2.5 py-1 text-xs font-medium rounded flex items-center gap-1.5 transition-colors ${
                    activeTool === 'probe'
                      ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/50'
                      : 'text-slate-400 hover:text-emerald-300'
                  }`}
                  title="Hover probe cursor to see neighborhood voting"
                >
                  <Radar className="w-3.5 h-3.5" />
                  <span>Probe Query</span>
                </button>
              )}
            </div>

            {/* Dataset Selector Dropdown */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                <Database className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <span className="hidden md:inline font-medium text-slate-300">Dataset:</span>
              </div>
              <div className="relative">
                <select
                  value={datasetPreset}
                  onChange={(e) => handleSetPreset(e.target.value as DatasetPreset)}
                  className="bg-slate-900 border border-slate-700/80 text-xs font-semibold text-slate-200 rounded-lg pl-3 pr-8 py-1.5 focus:outline-none focus:ring-1 focus:ring-cyan-500 cursor-pointer appearance-none hover:bg-slate-800 transition-colors shadow-sm"
                  title="Select synthetic data distribution"
                >
                  {DATASET_PRESETS.map((p) => (
                    <option key={p.id} value={p.id} className="bg-slate-900 text-slate-100 py-1">
                      {p.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>

              {/* Custom Ingested Dataset Indicator */}
              {customDatasetName && (
                <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 bg-emerald-950/60 border border-emerald-800/80 rounded-lg text-xs">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="font-semibold text-emerald-300 truncate max-w-[120px]">{customDatasetName}</span>
                  <span className="text-[10px] text-emerald-400/80 font-mono">({dots.length})</span>
                </div>
              )}

              {/* Import Custom Data Button */}
              {onOpenIngestModal && (
                <button
                  onClick={onOpenIngestModal}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 transition-colors shadow-sm whitespace-nowrap"
                  title="Import your data in any format (CSV, JSON, TSV, or AI Ingest)"
                >
                  <UploadCloud className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Import Data</span>
                </button>
              )}
            </div>

            {/* Playback Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg shadow-sm transition-all ${
                  isPlaying
                    ? 'bg-amber-600 hover:bg-amber-500 text-white'
                    : 'bg-cyan-600 hover:bg-cyan-500 text-white'
                }`}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                <span>{isPlaying ? 'Pause' : 'Run Simulation'}</span>
              </button>

              <button
                onClick={stepSimulation}
                className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors"
                title="Single Step Forward"
              >
                <SkipForward className="w-4 h-4" />
              </button>

              <button
                onClick={() => handleSetPreset(datasetPreset)}
                className="p-1.5 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors"
                title="Reset Dots"
              >
                <RotateCcw className="w-4 h-4" />
              </button>

              {/* Simulation Speed Slider Control */}
              <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
                <Gauge className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <span className="hidden md:inline text-[11px] font-medium text-slate-400">Speed:</span>
                <input
                  type="range"
                  min="0.2"
                  max="4"
                  step="0.1"
                  value={simSpeed}
                  onChange={(e) => setSimSpeed(Number(e.target.value))}
                  className="w-16 sm:w-20 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                  title={`Animation Speed: ${simSpeed.toFixed(1)}x`}
                />
                <span className="text-xs font-mono font-semibold text-cyan-400 tabular-nums w-7">
                  {simSpeed.toFixed(1)}x
                </span>
              </div>

              {onOpenChat && (
                <button
                  onClick={onOpenChat}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-indigo-950/80 hover:bg-indigo-900/80 text-indigo-300 border border-indigo-700/60 shadow-sm transition-all ml-1"
                  title="Ask Gemini AI Mentor about this simulation"
                >
                  <Bot className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="hidden sm:inline">AI Mentor</span>
                </button>
              )}
            </div>
          </div>

          {/* Canvas Rendering Plane */}
          <div className="flex-1 relative">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              className="w-full h-full block cursor-crosshair"
            />

            {/* Floating Visual Hint Banner */}
            <div className="absolute bottom-4 left-4 z-10 px-3 py-1.5 bg-slate-900/90 border border-slate-800 rounded-lg text-xs text-slate-300 backdrop-blur pointer-events-none">
              <span className="text-cyan-400 font-semibold mr-1.5">Interactive Tip:</span>
              Click any dot to inspect its math, or drag dots to watch the algorithm adapt in real
              time.
            </div>
          </div>
        </div>

        {/* Right Side Control Deck & Parameters (35% width) */}
        <div className="w-96 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0 overflow-y-auto">
          {/* Header */}
          <div className="p-5 border-b border-slate-800">
            <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
              <span className="font-semibold text-cyan-400">{currentAlgo.category}</span>
              <span aria-hidden="true">·</span>
              <span>Visual Playground</span>
            </div>
            <h3 className="text-xl font-bold text-white tracking-tight">{currentAlgo.name}</h3>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">{currentAlgo.tagline}</p>
          </div>

          {/* Real-Time Quantitative Metrics Dashboard with Animated Counters */}
          <div className="p-4 border-b border-slate-800">
            <MetricsDashboard algorithmId={algorithmId} metrics={currentMetrics} />
          </div>

          {/* Interactive Parameters Sliders Section */}
          <div className="p-5 border-b border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Parameter Controls
              </div>
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            </div>

            {/* Animation & Simulation Speed Slider Card */}
            <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 font-medium text-slate-300">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Animation & Step Speed</span>
                </span>
                <span className="font-mono text-cyan-400 font-bold tabular-nums">
                  {simSpeed.toFixed(1)}x
                </span>
              </div>

              <input
                type="range"
                min="0.2"
                max="4.0"
                step="0.1"
                value={simSpeed}
                onChange={(e) => setSimSpeed(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
              />

              <div className="flex items-center justify-between gap-1 pt-0.5">
                {[0.5, 1.0, 2.0, 4.0].map((spd) => (
                  <button
                    key={spd}
                    type="button"
                    onClick={() => setSimSpeed(spd)}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                      Math.abs(simSpeed - spd) < 0.05
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                        : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                    }`}
                  >
                    {spd.toFixed(1)}x
                  </button>
                ))}
              </div>
            </div>

            {/* Active Dataset Overview Badge */}
            {activePresetInfo && (
              <div className="p-3 bg-slate-950/70 rounded-lg border border-slate-800 text-xs space-y-1">
                <div className="flex items-center justify-between text-slate-400 text-[10px]">
                  <span className="flex items-center gap-1 font-semibold uppercase tracking-wider text-slate-300">
                    <Database className="w-3 h-3 text-cyan-400" />
                    <span>Active Distribution</span>
                  </span>
                  <span className="font-mono text-cyan-400">{activePresetInfo.dotCount} dots</span>
                </div>
                <div className="font-semibold text-slate-200">{activePresetInfo.name}</div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {activePresetInfo.description}
                </p>
              </div>
            )}

            {/* Specific Algorithm Sliders */}
            {algorithmId === 'kmeans' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Number of Centroids (K):</span>
                    <span className="font-mono text-cyan-400 font-bold">{params.kmeansK}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    step="1"
                    value={params.kmeansK}
                    onChange={(e) => setParams({ ...params, kmeansK: Number(e.target.value) })}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>K=2</span>
                    <span>K=4</span>
                    <span>K=6</span>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1.5">Seeding Technique:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {(['kmeans++', 'random'] as const).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setParams({ ...params, kmeansInit: mode })}
                        className={`py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                          params.kmeansInit === mode
                            ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {mode === 'kmeans++' ? 'K-Means++ (Smart)' : 'Random Seeds'}
                      </button>
                    ))}
                  </div>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.showVoronoi}
                    onChange={(e) => setParams({ ...params, showVoronoi: e.target.checked })}
                    className="rounded accent-cyan-500"
                  />
                  <span>Render Voronoi Territory Cells</span>
                </label>
              </div>
            )}

            {algorithmId === 'knn' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Neighbors to Consult (K):</span>
                    <span className="font-mono text-amber-400 font-bold">{params.knnK}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="2"
                    value={params.knnK}
                    onChange={(e) => setParams({ ...params, knnK: Number(e.target.value) })}
                    className="w-full accent-amber-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>K=1 (Sharp)</span>
                    <span>K=7</span>
                    <span>K=15 (Smooth)</span>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1.5">Voting Weight:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {(['uniform', 'distance'] as const).map((w) => (
                      <button
                        key={w}
                        onClick={() => setParams({ ...params, knnWeighting: w })}
                        className={`py-1.5 text-xs font-medium rounded-lg border transition-colors capitalize ${
                          params.knnWeighting === w
                            ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {w === 'uniform' ? '1 Vote Each' : 'Inverse Distance (1/d)'}
                      </button>
                    ))}
                  </div>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.showDecisionBoundary}
                    onChange={(e) =>
                      setParams({ ...params, showDecisionBoundary: e.target.checked })
                    }
                    className="rounded accent-amber-500"
                  />
                  <span>Display 2D Decision Heatmap</span>
                </label>
              </div>
            )}

            {algorithmId === 'linear-regression' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Gradient Descent Learning Rate (α):</span>
                    <span className="font-mono text-blue-400 font-bold">
                      {params.learningRate.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.01"
                    max="0.35"
                    step="0.01"
                    value={params.learningRate}
                    onChange={(e) =>
                      setParams({ ...params, learningRate: Number(e.target.value) })
                    }
                    className="w-full accent-blue-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>0.01 (Slow)</span>
                    <span>0.15 (Balanced)</span>
                    <span>0.35 (Aggressive)</span>
                  </div>
                </div>

                {/* 3D Loss Bowl Surface Mini-Visualization */}
                <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800">
                  <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                    <span className="font-medium text-slate-300">Loss Valley Surface (MSE)</span>
                    <span className="text-[10px] text-emerald-400">● Global Min</span>
                  </div>
                  <canvas
                    ref={lossCanvasRef}
                    width={320}
                    height={140}
                    className="w-full h-28 block bg-slate-950 rounded border border-slate-800/80"
                  />
                  <p className="text-[11px] text-slate-400 mt-2">
                    The red ball represents model parameters (θ₀, θ₁) rolling down the parabolic
                    loss bowl toward minimum error.
                  </p>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.showResiduals}
                    onChange={(e) => setParams({ ...params, showResiduals: e.target.checked })}
                    className="rounded accent-blue-500"
                  />
                  <span>Show Residual Error Sticks (e = y - ŷ)</span>
                </label>
              </div>
            )}

            {algorithmId === 'svm' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Regularization Penalty (C):</span>
                    <span className="font-mono text-violet-400 font-bold">{params.svmC.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="8.0"
                    step="0.2"
                    value={params.svmC}
                    onChange={(e) => setParams({ ...params, svmC: Number(e.target.value) })}
                    className="w-full accent-violet-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>C=0.2 (Wide / Soft)</span>
                    <span>C=4.0</span>
                    <span>C=8.0 (Strict / Hard)</span>
                  </div>
                </div>

                <div className="p-3 bg-violet-950/20 border border-violet-800/40 rounded-lg text-xs text-violet-200">
                  <p className="leading-relaxed">
                    <strong>Support Vector Principle:</strong> Only dots located on the margin
                    boundary (glowing purple rings) influence the hyperplane position. Interior dots
                    have zero influence on the decision boundary!
                  </p>
                </div>
              </div>
            )}

            {algorithmId === 'pca' && (
              <div className="space-y-3.5">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={params.showProjections}
                    onChange={(e) => setParams({ ...params, showProjections: e.target.checked })}
                    className="rounded accent-emerald-500"
                  />
                  <span>Draw Orthogonal Projections to PC1 Axis</span>
                </label>

                <div className="p-3 bg-emerald-950/20 border border-emerald-800/40 rounded-lg text-xs text-emerald-200">
                  <p className="leading-relaxed">
                    The solid green vector points in the direction of maximum dot spread. PCA
                    flattens high-dimensional dots onto this line while losing the least possible
                    variance.
                  </p>
                </div>
              </div>
            )}

            {algorithmId === 'decision-tree' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Maximum Tree Depth:</span>
                    <span className="font-mono text-orange-400 font-bold">{params.treeMaxDepth}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    step="1"
                    value={params.treeMaxDepth}
                    onChange={(e) =>
                      setParams({ ...params, treeMaxDepth: Number(e.target.value) })
                    }
                    className="w-full accent-orange-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Depth 1 (Stump)</span>
                    <span>Depth 2</span>
                    <span>Depth 4 (Complex)</span>
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1.5">Purity Criterion:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {(['gini', 'entropy'] as const).map((crit) => (
                      <button
                        key={crit}
                        onClick={() => setParams({ ...params, treeCriterion: crit })}
                        className={`py-1.5 text-xs font-medium rounded-lg border transition-colors capitalize ${
                          params.treeCriterion === crit
                            ? 'bg-orange-500/20 border-orange-500/50 text-orange-300'
                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {crit === 'gini' ? 'Gini Impurity' : 'Shannon Entropy'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {algorithmId === 'neural-net' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Hidden Layer Neurons:</span>
                    <span className="font-mono text-pink-400 font-bold">
                      {params.nnHiddenNeurons}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    step="1"
                    value={params.nnHiddenNeurons}
                    onChange={(e) =>
                      setParams({ ...params, nnHiddenNeurons: Number(e.target.value) })
                    }
                    className="w-full accent-pink-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1.5">Activation Non-Linearity:</div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {(['relu', 'sigmoid', 'tanh'] as const).map((act) => (
                      <button
                        key={act}
                        onClick={() => setParams({ ...params, nnActivation: act })}
                        className={`py-1 text-xs font-medium rounded border uppercase ${
                          params.nnActivation === act
                            ? 'bg-pink-500/20 border-pink-500/50 text-pink-300'
                            : 'bg-slate-950 border-slate-800 text-slate-400'
                        }`}
                      >
                        {act}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {algorithmId === 'dbscan' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Epsilon Radius (ε):</span>
                    <span className="font-mono text-teal-400 font-bold">{params.dbscanEps.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.30"
                    step="0.01"
                    value={params.dbscanEps}
                    onChange={(e) => setParams({ ...params, dbscanEps: Number(e.target.value) })}
                    className="w-full accent-teal-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>0.05 (Tight)</span>
                    <span>0.15</span>
                    <span>0.30 (Expansive)</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Minimum Points (MinPts):</span>
                    <span className="font-mono text-teal-400 font-bold">{params.dbscanMinPts}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    step="1"
                    value={params.dbscanMinPts}
                    onChange={(e) => setParams({ ...params, dbscanMinPts: Number(e.target.value) })}
                    className="w-full accent-teal-500 cursor-pointer"
                  />
                </div>

                <div className="p-3 bg-teal-950/20 border border-teal-800/40 rounded-lg text-xs text-teal-200">
                  <p className="leading-relaxed">
                    <strong>Core vs Noise:</strong> Dots with ≥ {params.dbscanMinPts} neighbors within radius {params.dbscanEps.toFixed(2)} form dense core clusters. Isolated dots become grey noise outliers.
                  </p>
                </div>
              </div>
            )}

            {algorithmId === 'logistic-regression' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Learning Rate (α):</span>
                    <span className="font-mono text-indigo-400 font-bold">{params.logRegLr.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.02"
                    max="0.4"
                    step="0.02"
                    value={params.logRegLr}
                    onChange={(e) => setParams({ ...params, logRegLr: Number(e.target.value) })}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Decision Threshold (P):</span>
                    <span className="font-mono text-indigo-400 font-bold">{params.logRegThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={params.logRegThreshold}
                    onChange={(e) => setParams({ ...params, logRegThreshold: Number(e.target.value) })}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>0.1 (High Recall)</span>
                    <span>0.5 (Default)</span>
                    <span>0.9 (High Precision)</span>
                  </div>
                </div>
              </div>
            )}

            {algorithmId === 'naive-bayes' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Variance Smoothing (σ²):</span>
                    <span className="font-mono text-yellow-400 font-bold">{params.nbVarSmoothing.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.001"
                    max="0.06"
                    step="0.002"
                    value={params.nbVarSmoothing}
                    onChange={(e) => setParams({ ...params, nbVarSmoothing: Number(e.target.value) })}
                    className="w-full accent-yellow-500 cursor-pointer"
                  />
                </div>

                <div className="p-3 bg-yellow-950/20 border border-yellow-800/40 rounded-lg text-xs text-yellow-200">
                  <p className="leading-relaxed">
                    <strong>Gaussian Likelihood Ellipses:</strong> The rings show 1-standard-deviation contours for feature distribution per class under the naive independence assumption.
                  </p>
                </div>
              </div>
            )}

            {algorithmId === 'random-forest' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Ensemble Trees (B):</span>
                    <span className="font-mono text-emerald-400 font-bold">{params.rfNumTrees}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="12"
                    step="1"
                    value={params.rfNumTrees}
                    onChange={(e) => setParams({ ...params, rfNumTrees: Number(e.target.value) })}
                    className="w-full accent-emerald-500 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>1 (Single Tree)</span>
                    <span>6</span>
                    <span>12 (Smooth Ensemble)</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Tree Max Depth:</span>
                    <span className="font-mono text-emerald-400 font-bold">{params.rfMaxDepth}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    step="1"
                    value={params.rfMaxDepth}
                    onChange={(e) => setParams({ ...params, rfMaxDepth: Number(e.target.value) })}
                    className="w-full accent-emerald-500 cursor-pointer"
                  />
                </div>
              </div>
            )}

            {algorithmId === 'hierarchical' && (
              <div className="space-y-3.5">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Target Clusters (K):</span>
                    <span className="font-mono text-rose-400 font-bold">{params.hierarchicalClusters}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    step="1"
                    value={params.hierarchicalClusters}
                    onChange={(e) => setParams({ ...params, hierarchicalClusters: Number(e.target.value) })}
                    className="w-full accent-rose-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="text-xs text-slate-300 mb-1.5">Linkage Metric:</div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {(['single', 'complete', 'average'] as const).map((lnk) => (
                      <button
                        key={lnk}
                        onClick={() => setParams({ ...params, hierarchicalLinkage: lnk })}
                        className={`py-1 text-xs font-medium rounded border capitalize ${
                          params.hierarchicalLinkage === lnk
                            ? 'bg-rose-500/20 border-rose-500/50 text-rose-300'
                            : 'bg-slate-950 border-slate-800 text-slate-400'
                        }`}
                      >
                        {lnk}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Mathematical Formula & Concept Explainer */}
          <div className="p-5 space-y-3">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Mathematical Foundation
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-xs text-cyan-300 overflow-x-auto">
              {currentAlgo.mathFormula}
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {currentAlgo.formulaExplanation}
            </p>
          </div>
        </div>

        {/* Selected Dot Inspector Drawer */}
        {selectedDot && (
          <DotInspector
            dot={selectedDot}
            algorithmId={algorithmId}
            onClose={() => setSelectedDot(null)}
            onDeleteDot={(id) => {
              setDots((prev) => prev.filter((d) => d.id !== id));
              setSelectedDot(null);
            }}
            extraInfo={{
              centroidDistances:
                algorithmId === 'kmeans'
                  ? centroids.map((c) => ({
                      id: c.id,
                      color: c.color,
                      dist: Math.hypot(selectedDot.x - c.x, selectedDot.y - c.y),
                    }))
                  : undefined,
              slope: gdState.theta1,
              intercept: gdState.theta0,
            }}
          />
        )}
      </div>
    </div>
  );
};
