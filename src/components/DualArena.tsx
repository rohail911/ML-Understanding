import React, { useState, useEffect, useRef } from 'react';
import { AlgorithmId, DataDot } from '../types/ml';
import { ALGORITHMS } from '../utils/algorithms';
import { generateDataset, DatasetPreset } from '../utils/datasets';
import {
  solveSVM,
  queryKNN,
  buildDecisionTreePartitions,
  solveLogisticRegression,
  fitGaussianNaiveBayes,
  trainRandomForest,
  runDBSCAN,
  runHierarchicalClustering,
  computeOLS,
  computePCA,
  runKMeansStep,
  initCentroids,
} from '../utils/math';
import { Layers, RotateCcw, Plus, Move, UploadCloud } from 'lucide-react';

interface DualArenaProps {
  customDots?: DataDot[];
  customDatasetName?: string;
  onOpenIngestModal?: () => void;
}

export const DualArena: React.FC<DualArenaProps> = ({
  customDots,
  customDatasetName,
  onOpenIngestModal,
}) => {
  const [algoLeft, setAlgoLeft] = useState<AlgorithmId>('svm');
  const [algoRight, setAlgoRight] = useState<AlgorithmId>('knn');
  const [preset, setPreset] = useState<DatasetPreset>('moons');
  const [dots, setDots] = useState<DataDot[]>(() => customDots || generateDataset('moons', 46, 0.05));

  // Sync custom dots if provided
  useEffect(() => {
    if (customDots && customDots.length > 0) {
      setDots(customDots);
    }
  }, [customDots]);

  const canvasLeftRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRightRef = useRef<HTMLCanvasElement | null>(null);
  const draggingDotIdRef = useRef<string | null>(null);

  const leftMeta = ALGORITHMS.find((a) => a.id === algoLeft) || ALGORITHMS[0];
  const rightMeta = ALGORITHMS.find((a) => a.id === algoRight) || ALGORITHMS[1];

  const handleResetData = (newPreset: DatasetPreset) => {
    setPreset(newPreset);
    setDots(generateDataset(newPreset, 46, 0.05));
  };

  // Render Left Canvas
  useEffect(() => {
    const canvas = canvasLeftRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    renderModelToCanvas(ctx, canvas.width, canvas.height, algoLeft, dots);
  }, [algoLeft, dots]);

  // Render Right Canvas
  useEffect(() => {
    const canvas = canvasRightRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    renderModelToCanvas(ctx, canvas.width, canvas.height, algoRight, dots);
  }, [algoRight, dots]);

  // Shared rendering logic for any of the 12 algorithms
  const renderModelToCanvas = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    algoId: AlgorithmId,
    dataDots: DataDot[]
  ) => {
    ctx.clearRect(0, 0, width, height);

    const toX = (nx: number) => nx * width;
    const toY = (ny: number) => (1 - ny) * height;

    // Draw grid
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 35) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 35) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Render Algorithm Decision Boundary or Model Mechanics
    if (algoId === 'svm') {
      const svmRes = solveSVM(dataDots, 1.0, 'linear');
      const [w1, w2] = svmRes.weights;
      const b = svmRes.bias;
      if (Math.abs(w2) > 0.001) {
        const getY = (nx: number, offset: number) => -(w1 * nx + b + offset) / w2;
        // Hyperplane
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(getY(0, 0)));
        ctx.lineTo(toX(1), toY(getY(1, 0)));
        ctx.stroke();

        // Margins
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(getY(0, 1)));
        ctx.lineTo(toX(1), toY(getY(1, 1)));
        ctx.moveTo(toX(0), toY(getY(0, -1)));
        ctx.lineTo(toX(1), toY(getY(1, -1)));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else if (algoId === 'knn') {
      const step = 20;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = (x + step / 2) / width;
          const ny = 1 - (y + step / 2) / height;
          const q = queryKNN(dataDots, nx, ny, 5, 'uniform');
          const p1 = q.probabilities[1] ?? 0.5;
          ctx.fillStyle =
            p1 > 0.5
              ? `rgba(245, 158, 11, ${(p1 - 0.5) * 0.35})`
              : `rgba(6, 182, 212, ${(0.5 - p1) * 0.35})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algoId === 'decision-tree') {
      const dt = buildDecisionTreePartitions(dataDots, 3, 'gini');
      dt.partitions.forEach((part) => {
        const px = toX(part.xMin);
        const py = toY(part.yMax);
        const pw = toX(part.xMax) - px;
        const ph = toY(part.yMin) - py;
        const color = part.predictedLabel === 1 ? '245, 158, 11' : '6, 182, 212';
        ctx.fillStyle = `rgba(${color}, 0.15)`;
        ctx.fillRect(px, py, pw, ph);
        ctx.strokeStyle = `rgba(${color}, 0.3)`;
        ctx.strokeRect(px, py, pw, ph);
      });
    } else if (algoId === 'logistic-regression') {
      const lr = solveLogisticRegression(dataDots, 0.5, 60, 0.5);
      const { w1, w2, b } = lr;
      if (Math.abs(w2) > 0.001) {
        const getY = (nx: number) => -(w1 * nx + b) / w2;
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(getY(0)));
        ctx.lineTo(toX(1), toY(getY(1)));
        ctx.stroke();
      }
    } else if (algoId === 'naive-bayes') {
      const nb = fitGaussianNaiveBayes(dataDots);
      const step = 20;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = (x + step / 2) / width;
          const ny = 1 - (y + step / 2) / height;
          const { prob0, prob1 } = nb.predictProb(nx, ny);
          ctx.fillStyle =
            prob1 > prob0
              ? `rgba(245, 158, 11, ${(prob1 - 0.5) * 0.3})`
              : `rgba(168, 85, 247, ${(prob0 - 0.5) * 0.3})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algoId === 'random-forest') {
      const rf = trainRandomForest(dataDots, 5, 3);
      const step = 22;
      for (let x = 0; x < width; x += step) {
        for (let y = 0; y < height; y += step) {
          const nx = (x + step / 2) / width;
          const ny = 1 - (y + step / 2) / height;
          const p1 = rf.predictProb(nx, ny);
          ctx.fillStyle =
            p1 > 0.5
              ? `rgba(245, 158, 11, ${(p1 - 0.5) * 0.35})`
              : `rgba(217, 119, 6, ${(0.5 - p1) * 0.35})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    } else if (algoId === 'linear-regression') {
      const lr = computeOLS(dataDots);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(lr.intercept));
      ctx.lineTo(toX(1), toY(lr.slope + lr.intercept));
      ctx.stroke();

      // Residual lines
      dataDots.forEach((d) => {
        const predY = lr.slope * d.x + lr.intercept;
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toX(d.x), toY(d.y));
        ctx.lineTo(toX(d.x), toY(predY));
        ctx.stroke();
      });
    } else if (algoId === 'kmeans') {
      let centroids = initCentroids(dataDots, 3, 'kmeans++');
      for (let i = 0; i < 4; i++) {
        const step = runKMeansStep(dataDots, centroids);
        centroids = step.centroids;
      }
      centroids.forEach((c) => {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(toX(c.x), toY(c.y), 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    } else if (algoId === 'dbscan') {
      const db = runDBSCAN(dataDots, 0.12, 4);
      db.dots.forEach((d) => {
        if (d.isNoise) {
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.6)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(toX(d.x), toY(d.y), 10, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    } else if (algoId === 'pca') {
      const pca = computePCA(dataDots);
      const cx = pca.mean.x;
      const cy = pca.mean.y;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(toX(cx - pca.pc1.x * 0.4), toY(cy - pca.pc1.y * 0.4));
      ctx.lineTo(toX(cx + pca.pc1.x * 0.4), toY(cy + pca.pc1.y * 0.4));
      ctx.stroke();
    }

    // Render Dots
    dataDots.forEach((d) => {
      const cx = toX(d.x);
      const cy = toY(d.y);
      const color = d.label === 1 ? '#f59e0b' : '#06b6d4';

      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / canvas.width;
    const ny = 1 - (e.clientY - rect.top) / canvas.height;

    let closest: DataDot | null = null;
    let minD = 0.05;
    dots.forEach((d) => {
      const dist = Math.hypot(d.x - nx, d.y - ny);
      if (dist < minD) {
        minD = dist;
        closest = d;
      }
    });

    if (closest) {
      draggingDotIdRef.current = (closest as DataDot).id;
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingDotIdRef.current) return;
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const nx = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / canvas.width));
    const ny = Math.max(0.02, Math.min(0.98, 1 - (e.clientY - rect.top) / canvas.height));

    setDots((prev) =>
      prev.map((d) => (d.id === draggingDotIdRef.current ? { ...d, x: nx, y: ny } : d))
    );
  };

  const handleCanvasMouseUp = () => {
    draggingDotIdRef.current = null;
  };

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-61px)] bg-slate-950 p-6 overflow-y-auto">
      {/* Top Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-slate-900 border border-slate-800 rounded-xl mb-6 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white tracking-tight">Dual Algorithm Arena</h2>
            <div className="text-xs text-slate-400">
              Manipulate dots once and observe how two completely different algorithms interpret them!
            </div>
          </div>
        </div>

        {/* Data Distribution Buttons */}
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-400 font-medium mr-1">Dataset:</span>
          {(['moons', 'circles', 'blobs', 'spiral', 'linear', 'xor', 'outliers'] as DatasetPreset[]).map((p) => (
            <button
              key={p}
              onClick={() => handleResetData(p)}
              className={`px-2.5 py-1 text-xs font-semibold rounded-lg capitalize transition-colors ${
                preset === p
                  ? 'bg-cyan-500/20 border border-cyan-500/50 text-cyan-300'
                  : 'bg-slate-950 border border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {p}
            </button>
          ))}
          <button
            onClick={() => handleResetData(preset)}
            className="p-1.5 text-slate-400 hover:text-white bg-slate-950 border border-slate-800 rounded-lg ml-1 transition-colors"
            title="Randomize dots"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Custom Dataset indicator */}
          {customDatasetName && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-950/60 border border-emerald-800/80 rounded-lg text-xs ml-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-semibold text-emerald-300 truncate max-w-[120px]">{customDatasetName}</span>
              <span className="text-[10px] text-emerald-400/80 font-mono">({dots.length})</span>
            </div>
          )}

          {/* Import Data button */}
          {onOpenIngestModal && (
            <button
              onClick={onOpenIngestModal}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 ml-1 transition-colors shadow-sm"
              title="Import any dataset format"
            >
              <UploadCloud className="w-3.5 h-3.5 text-emerald-400" />
              <span>Import Data</span>
            </button>
          )}
        </div>
      </div>

      {/* Side-by-Side Dual View */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-[460px]">
        {/* Model Left */}
        <div className="flex flex-col bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: leftMeta.color }}
              />
              <span className="text-sm font-bold text-white">{leftMeta.name}</span>
            </div>
            <select
              value={algoLeft}
              onChange={(e) => setAlgoLeft(e.target.value as AlgorithmId)}
              className="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1 font-medium"
            >
              {ALGORITHMS.map((algo) => (
                <option key={algo.id} value={algo.id}>
                  {algo.name} ({algo.category})
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 relative aspect-4/3 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <canvas
              ref={canvasLeftRef}
              width={500}
              height={380}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              className="w-full h-full block cursor-grab active:cursor-grabbing"
            />
          </div>

          <div className="mt-3 text-xs text-slate-400">
            <strong>Behavior:</strong> {leftMeta.tagline}
          </div>
        </div>

        {/* Model Right */}
        <div className="flex flex-col bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: rightMeta.color }}
              />
              <span className="text-sm font-bold text-white">{rightMeta.name}</span>
            </div>
            <select
              value={algoRight}
              onChange={(e) => setAlgoRight(e.target.value as AlgorithmId)}
              className="bg-slate-950 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1 font-medium"
            >
              {ALGORITHMS.map((algo) => (
                <option key={algo.id} value={algo.id}>
                  {algo.name} ({algo.category})
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 relative aspect-4/3 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <canvas
              ref={canvasRightRef}
              width={500}
              height={380}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              className="w-full h-full block cursor-grab active:cursor-grabbing"
            />
          </div>

          <div className="mt-3 text-xs text-slate-400">
            <strong>Behavior:</strong> {rightMeta.tagline}
          </div>
        </div>
      </div>
    </div>
  );
};
