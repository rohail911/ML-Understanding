import React from 'react';
import { DataDot, AlgorithmId } from '../types/ml';
import { ALGORITHMS } from '../utils/algorithms';
import { X, Target, Move, HelpCircle, CheckCircle2 } from 'lucide-react';

interface DotInspectorProps {
  dot: DataDot | null;
  algorithmId: AlgorithmId;
  onClose: () => void;
  onDeleteDot?: (id: string) => void;
  extraInfo?: any;
}

export const DotInspector: React.FC<DotInspectorProps> = ({
  dot,
  algorithmId,
  onClose,
  onDeleteDot,
  extraInfo,
}) => {
  if (!dot) return null;

  const algo = ALGORITHMS.find((a) => a.id === algorithmId);

  // Generate algorithm-specific insights
  const renderAlgorithmInsights = () => {
    switch (algorithmId) {
      case 'kmeans': {
        const centroidDistances = extraInfo?.centroidDistances as
          | Array<{ id: number; dist: number; color: string }>
          | undefined;

        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Clustering Membership</div>
              <div className="text-white font-semibold flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor:
                      centroidDistances?.find((c) => c.id === dot.cluster)?.color || '#06b6d4',
                  }}
                />
                <span>Assigned to Cluster #{dot.cluster !== undefined ? dot.cluster + 1 : '—'}</span>
              </div>
            </div>

            {centroidDistances && centroidDistances.length > 0 && (
              <div>
                <div className="text-[11px] font-medium text-slate-400 mb-1.5">
                  Euclidean Distances to Centroids:
                </div>
                <div className="space-y-1.5 font-mono text-xs">
                  {centroidDistances.map((c) => {
                    const isWinner = c.id === dot.cluster;
                    return (
                      <div
                        key={c.id}
                        className={`flex items-center justify-between p-2 rounded ${
                          isWinner
                            ? 'bg-cyan-950/40 border border-cyan-800/60 text-cyan-200'
                            : 'bg-slate-900 border border-slate-800 text-slate-400'
                        }`}
                      >
                        <span className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: c.color }}
                          />
                          <span>Centroid {c.id + 1}</span>
                        </span>
                        <span className="tabular-nums">{c.dist.toFixed(4)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            <p className="text-xs text-slate-400 leading-relaxed">
              In K-Means, this dot is owned by the centroid with the smallest Euclidean distance.
              When centroids recalculate, this dot pulls its assigned centroid toward its coordinates.
            </p>
          </div>
        );
      }

      case 'knn': {
        const isNeighbor = extraInfo?.isNeighbor;
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Ground Truth Class</div>
              <div className="text-white font-semibold flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: dot.label === 1 ? '#f59e0b' : '#06b6d4' }}
                />
                <span>Class {dot.label}</span>
                {isNeighbor && (
                  <span className="ml-auto text-emerald-400 flex items-center gap-1 text-[11px]">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Active Voting Neighbor</span>
                  </span>
                )}
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              This dot acts as a voting training exemplar. When any query cursor hovers nearby, this
              dot casts its vote for Class {dot.label}.
            </p>
          </div>
        );
      }

      case 'linear-regression': {
        const slope = extraInfo?.slope ?? 0.7;
        const intercept = extraInfo?.intercept ?? 0.15;
        const predictedY = slope * dot.x + intercept;
        const residual = dot.y - predictedY;

        return (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-slate-400 text-[10px] font-sans">Observed Target (Y)</div>
                <div className="text-white font-semibold tabular-nums mt-0.5">{dot.y.toFixed(4)}</div>
              </div>
              <div className="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-slate-400 text-[10px] font-sans">Fitted Model (Ŷ)</div>
                <div className="text-cyan-300 font-semibold tabular-nums mt-0.5">
                  {predictedY.toFixed(4)}
                </div>
              </div>
            </div>

            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="flex justify-between items-center text-slate-300 mb-1">
                <span>Residual Error (e = y - ŷ):</span>
                <span
                  className={`font-mono tabular-nums font-semibold ${
                    residual > 0 ? 'text-amber-400' : 'text-rose-400'
                  }`}
                >
                  {residual.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between items-center text-slate-400">
                <span>Squared Loss Contribution:</span>
                <span className="font-mono tabular-nums text-white">
                  {(residual * residual).toFixed(5)}
                </span>
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              The red vertical line connecting this dot to the regression line represents its
              residual error. Gradient descent adjusts the line to minimize the sum of all squared
              residuals.
            </p>
          </div>
        );
      }

      case 'svm': {
        const isSV = dot.isSupportVector;
        return (
          <div className="space-y-3">
            <div
              className={`p-3 rounded-lg border text-xs ${
                isSV
                  ? 'bg-violet-950/50 border-violet-700/80 text-violet-200'
                  : 'bg-slate-950/60 border-slate-800 text-slate-300'
              }`}
            >
              <div className="font-semibold text-sm mb-1 flex items-center justify-between">
                <span>{isSV ? 'Active Support Vector' : 'Interior Data Dot'}</span>
                {isSV && <span className="text-[10px] bg-violet-500/30 px-1.5 py-0.5 rounded">Critical Boundary</span>}
              </div>
              <p className="text-xs leading-relaxed opacity-90">
                {isSV
                  ? 'This dot touches or violates the margin corridor. Removing this dot would directly tilt or shift the optimal hyperplane!'
                  : 'This dot sits comfortably inside its class territory. Removing it has ZERO effect on the separating hyperplane.'}
              </p>
            </div>
          </div>
        );
      }

      case 'pca': {
        const score = extraInfo?.score ?? 0;
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 mb-1">Projection Coordinate (PC1 Score)</div>
              <div className="text-emerald-400 font-mono text-base font-bold tabular-nums">
                {score.toFixed(4)}
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              When dimensionality is reduced from 2D to 1D, this dot’s position collapses to this
              single scalar value along the primary eigenvector.
            </p>
          </div>
        );
      }

      case 'decision-tree': {
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Tree Leaf Assignment</div>
              <div className="text-white font-semibold flex items-center justify-between">
                <span>Class {dot.label} Exemplar</span>
                <span className="text-[10px] bg-orange-500/20 text-orange-300 px-1.5 py-0.5 rounded font-mono">
                  Orthogonal Partition
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Decision trees recursively slice space along axis-aligned planes. This dot is contained
              within a specific rectangular leaf box chosen to minimize Gini impurity.
            </p>
          </div>
        );
      }

      case 'neural-net': {
        const label = dot.label ?? 0;
        const prob = extraInfo?.prediction ?? (label === 1 ? 0.88 : 0.12);
        const ceLoss = -(label * Math.log(Math.max(1e-5, prob)) + (1 - label) * Math.log(Math.max(1e-5, 1 - prob)));
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Sigmoid Activation & Loss</div>
              <div className="flex justify-between items-center text-sm font-bold text-white font-mono">
                <span>P(y=1): {(prob * 100).toFixed(1)}%</span>
                <span className="text-xs font-normal text-pink-400">Loss: {ceLoss.toFixed(4)}</span>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Features (x₁, x₂) pass through hidden layers with non-linear activation. Backpropagation
              nudges all synapse weights in direction of negative gradient for this dot.
            </p>
          </div>
        );
      }

      case 'dbscan': {
        const isNoise = dot.isNoise ?? false;
        const isCore = dot.isCore ?? (dot.cluster !== undefined && dot.cluster >= 0 && !isNoise);
        return (
          <div className="space-y-3">
            <div className={`p-3 rounded-lg border text-xs ${
              isNoise 
                ? 'bg-rose-950/40 border-rose-800/60 text-rose-200'
                : isCore 
                ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200'
                : 'bg-cyan-950/40 border-cyan-800/60 text-cyan-200'
            }`}>
              <div className="font-semibold text-sm mb-1 flex items-center justify-between">
                <span>{isNoise ? 'Noise Outlier Dot' : isCore ? 'Density Core Dot' : 'Border Cluster Dot'}</span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-900/60">
                  {isNoise ? 'Unassigned' : `Cluster #${(dot.cluster ?? 0) + 1}`}
                </span>
              </div>
              <p className="text-xs leading-relaxed opacity-90">
                {isNoise
                  ? 'This dot does not have minPts within radius ε and is not reachable from any core dot. DBSCAN flags it as noise anomaly!'
                  : isCore
                  ? 'This dot has ≥ minPts neighbors inside radius ε, forming a high-density core that expands the cluster.'
                  : 'This dot sits on the boundary of a density cluster, connected to at least one core dot within radius ε.'}
              </p>
            </div>
          </div>
        );
      }

      case 'logistic-regression': {
        const prob = extraInfo?.probability ?? (1 / (1 + Math.exp(-(3 * dot.x + 2 * dot.y - 2.5))));
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs font-mono">
              <div className="text-slate-400 mb-1 font-sans">Posterior Sigmoid Probability σ(z)</div>
              <div className="flex justify-between items-center">
                <span className="text-cyan-300 font-bold text-base tabular-nums">{(prob * 100).toFixed(2)}%</span>
                <span className="text-xs text-slate-400 font-sans">Class {prob >= 0.5 ? 1 : 0}</span>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Logistic regression maps linear combination z = w·x + b through the S-shaped sigmoid function
              σ(z) = 1 / (1 + e⁻ᶻ) to output a calibrated probability between 0 and 1.
            </p>
          </div>
        );
      }

      case 'naive-bayes': {
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Gaussian Likelihood Evaluation</div>
              <div className="text-white font-semibold">Class {dot.label} Data Point</div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Assumes feature independence: P(x₁, x₂ | C) = P(x₁|C) · P(x₂|C). Evaluated under the
              fitted 2D Gaussian bell curves for each class.
            </p>
          </div>
        );
      }

      case 'random-forest': {
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Ensemble Bagging Consensus</div>
              <div className="text-amber-300 font-semibold flex items-center justify-between">
                <span>Multi-Tree Aggregate</span>
                <span className="text-[10px] bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded font-mono">
                  Majority Vote
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Multiple bootstrap decision trees evaluate this dot independently. Majority voting
              cancels out variance and creates smooth, robust non-linear boundaries.
            </p>
          </div>
        );
      }

      case 'hierarchical': {
        return (
          <div className="space-y-3">
            <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs">
              <div className="text-slate-400 font-medium mb-1">Dendrogram Tree Branch</div>
              <div className="text-teal-300 font-semibold">
                Cluster #{dot.cluster !== undefined ? dot.cluster + 1 : 1}
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Agglomerative clustering started with this dot as its own individual singleton cluster
              and progressively merged it with its closest neighbor based on the chosen linkage metric.
            </p>
          </div>
        );
      }

      default:
        return (
          <p className="text-xs text-slate-400 leading-relaxed">
            Every dot represents a data sample with features (X, Y) contributing to the model's
            predictions.
          </p>
        );
    }
  };

  return (
    <div className="w-80 bg-slate-900 border-l border-slate-800 p-5 flex flex-col justify-between shrink-0 overflow-y-auto">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
              <Target className="w-3.5 h-3.5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white tracking-tight">Dot Inspector</h4>
              <div className="text-[10px] text-slate-400 font-mono">{dot.id}</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Feature Coordinates */}
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div className="p-2.5 bg-slate-950/70 border border-slate-800 rounded-lg">
            <span className="text-slate-500 block text-[10px] font-sans">Feature X₁</span>
            <span className="text-white font-semibold tabular-nums">{dot.x.toFixed(4)}</span>
          </div>
          <div className="p-2.5 bg-slate-950/70 border border-slate-800 rounded-lg">
            <span className="text-slate-500 block text-[10px] font-sans">Feature X₂ (or Y)</span>
            <span className="text-white font-semibold tabular-nums">{dot.y.toFixed(4)}</span>
          </div>
        </div>

        {/* Algorithm Insights */}
        <div>{renderAlgorithmInsights()}</div>

        {/* Interaction Callout */}
        <div className="p-3 bg-cyan-950/20 border border-cyan-800/40 rounded-lg flex items-start gap-2.5 text-xs text-cyan-200">
          <Move className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <p className="leading-snug">
            <strong className="text-cyan-300">Live Drag:</strong> You can drag this dot directly on
            the canvas and watch how {algo?.name || 'the model'} dynamically recalculates in real
            time!
          </p>
        </div>
      </div>

      {/* Footer Actions */}
      {onDeleteDot && (
        <div className="pt-4 border-t border-slate-800 mt-4">
          <button
            onClick={() => onDeleteDot(dot.id)}
            className="w-full py-1.5 px-3 text-xs font-medium text-rose-400 hover:text-rose-300 hover:bg-rose-950/30 border border-rose-900/40 rounded-lg transition-colors"
          >
            Remove This Dot
          </button>
        </div>
      )}
    </div>
  );
};
