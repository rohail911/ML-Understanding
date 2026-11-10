import { DataDot } from '../types/ml';

export type DatasetPreset =
  | 'blobs'
  | 'moons'
  | 'circles'
  | 'linear'
  | 'spiral'
  | 'xor'
  | 'outliers';

export interface DatasetPresetInfo {
  id: DatasetPreset;
  name: string;
  category: string;
  description: string;
  dotCount: number;
}

export const DATASET_PRESETS: DatasetPresetInfo[] = [
  {
    id: 'blobs',
    name: 'Gaussian Clusters (Blobs)',
    category: 'Clustering & Classification',
    description: 'Distinct Gaussian distribution clouds ideal for K-Means and linear separation.',
    dotCount: 54,
  },
  {
    id: 'moons',
    name: 'Two Interlocking Moons',
    category: 'Non-Linear Geometry',
    description: 'Non-linear crescents where linear hyperplanes fail and local methods (KNN, kernels) excel.',
    dotCount: 50,
  },
  {
    id: 'circles',
    name: 'Concentric Circles',
    category: 'Radial Geometry',
    description: 'An inner circle enveloped by an outer ring. Demonstrates why RBF kernels or neural nets are needed.',
    dotCount: 50,
  },
  {
    id: 'linear',
    name: 'Linear Trend with Noise',
    category: 'Regression & Fitting',
    description: 'Continuous linear correlation for studying gradient descent slopes and residual errors.',
    dotCount: 42,
  },
  {
    id: 'spiral',
    name: 'Dual Archimedean Spirals',
    category: 'Extreme Non-Linear',
    description: 'Two interlocking spiral arms challenging decision boundaries to wrap without self-intersecting.',
    dotCount: 56,
  },
  {
    id: 'xor',
    name: 'XOR Diagonal Quadrants',
    category: 'Non-Linearly Separable',
    description: 'Classic XOR distribution requiring multi-depth trees or hidden layer feature synthesis.',
    dotCount: 48,
  },
  {
    id: 'outliers',
    name: 'Linear Trend with Outliers',
    category: 'Robustness & Sensitivity',
    description: 'Linear data contaminated by severe adversarial outliers to test regularized leverage and loss.',
    dotCount: 45,
  },
];

export function generateDataset(preset: DatasetPreset, count = 60, noise = 0.05): DataDot[] {
  const dots: DataDot[] = [];

  switch (preset) {
    case 'blobs': {
      // 3 natural clusters
      const centers = [
        { x: 0.25, y: 0.35, label: 0 },
        { x: 0.75, y: 0.35, label: 1 },
        { x: 0.5, y: 0.75, label: 0 },
      ];
      const perCenter = Math.floor(count / centers.length);
      let id = 1;
      centers.forEach((center) => {
        for (let i = 0; i < perCenter; i++) {
          const u1 = Math.random();
          const u2 = Math.random();
          const r = Math.sqrt(-2 * Math.log(u1 || 0.0001)) * (0.08 + noise);
          const theta = 2 * Math.PI * u2;
          const x = Math.max(0.05, Math.min(0.95, center.x + r * Math.cos(theta)));
          const y = Math.max(0.05, Math.min(0.95, center.y + r * Math.sin(theta)));
          dots.push({
            id: `dot-${id++}`,
            x,
            y,
            label: center.label,
            continuousTarget: y,
          });
        }
      });
      break;
    }

    case 'moons': {
      // Classic two interlocking semi-circles
      const half = Math.floor(count / 2);
      let id = 1;
      // Moon 0 (Upper)
      for (let i = 0; i < half; i++) {
        const phi = (i / half) * Math.PI;
        const x = 0.35 + 0.28 * Math.cos(phi) + (Math.random() - 0.5) * noise * 1.5;
        const y = 0.38 + 0.22 * Math.sin(phi) + (Math.random() - 0.5) * noise * 1.5;
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.05, Math.min(0.95, x)),
          y: Math.max(0.05, Math.min(0.95, y)),
          label: 0,
          continuousTarget: y,
        });
      }
      // Moon 1 (Lower)
      for (let i = 0; i < half; i++) {
        const phi = (i / half) * Math.PI;
        const x = 0.65 - 0.28 * Math.cos(phi) + (Math.random() - 0.5) * noise * 1.5;
        const y = 0.62 - 0.22 * Math.sin(phi) + (Math.random() - 0.5) * noise * 1.5;
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.05, Math.min(0.95, x)),
          y: Math.max(0.05, Math.min(0.95, y)),
          label: 1,
          continuousTarget: y,
        });
      }
      break;
    }

    case 'circles': {
      // Inner circle vs Outer ring
      const innerCount = Math.floor(count * 0.4);
      const outerCount = count - innerCount;
      let id = 1;

      // Inner
      for (let i = 0; i < innerCount; i++) {
        const r = 0.15 * Math.sqrt(Math.random());
        const theta = Math.random() * 2 * Math.PI;
        const x = 0.5 + r * Math.cos(theta) + (Math.random() - 0.5) * noise;
        const y = 0.5 + r * Math.sin(theta) + (Math.random() - 0.5) * noise;
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.05, Math.min(0.95, x)),
          y: Math.max(0.05, Math.min(0.95, y)),
          label: 0,
          continuousTarget: y,
        });
      }

      // Outer ring
      for (let i = 0; i < outerCount; i++) {
        const r = 0.35 + (Math.random() - 0.5) * 0.08;
        const theta = Math.random() * 2 * Math.PI;
        const x = 0.5 + r * Math.cos(theta) + (Math.random() - 0.5) * noise;
        const y = 0.5 + r * Math.sin(theta) + (Math.random() - 0.5) * noise;
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.05, Math.min(0.95, x)),
          y: Math.max(0.05, Math.min(0.95, y)),
          label: 1,
          continuousTarget: y,
        });
      }
      break;
    }

    case 'linear': {
      // Linear correlation: y = m*x + c + noise
      let id = 1;
      const slope = 0.7;
      const intercept = 0.15;
      for (let i = 0; i < count; i++) {
        const x = 0.1 + (i / count) * 0.8 + (Math.random() - 0.5) * 0.05;
        const y = slope * x + intercept + (Math.random() - 0.5) * (noise * 3);
        const clampedY = Math.max(0.05, Math.min(0.95, y));
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.05, Math.min(0.95, x)),
          y: clampedY,
          label: clampedY > 0.5 ? 1 : 0,
          continuousTarget: clampedY,
        });
      }
      break;
    }

    case 'spiral': {
      // Two interleaving spirals
      const half = Math.floor(count / 2);
      let id = 1;
      for (let arm = 0; arm < 2; arm++) {
        for (let i = 0; i < half; i++) {
          const t = i / half;
          const r = 0.08 + t * 0.38;
          const theta = t * 3.5 + arm * Math.PI;
          const x = 0.5 + r * Math.cos(theta) + (Math.random() - 0.5) * noise;
          const y = 0.5 + r * Math.sin(theta) + (Math.random() - 0.5) * noise;
          dots.push({
            id: `dot-${id++}`,
            x: Math.max(0.05, Math.min(0.95, x)),
            y: Math.max(0.05, Math.min(0.95, y)),
            label: arm,
            continuousTarget: y,
          });
        }
      }
      break;
    }

    case 'xor': {
      // 4 quadrants: Top-left and bottom-right are Class 0; top-right and bottom-left are Class 1
      const quarter = Math.floor(count / 4);
      let id = 1;
      const quadrants = [
        { cx: 0.28, cy: 0.72, label: 0 },
        { cx: 0.72, cy: 0.72, label: 1 },
        { cx: 0.28, cy: 0.28, label: 1 },
        { cx: 0.72, cy: 0.28, label: 0 },
      ];
      quadrants.forEach((q) => {
        for (let i = 0; i < quarter; i++) {
          const rx = (Math.random() - 0.5) * 0.28;
          const ry = (Math.random() - 0.5) * 0.28;
          const x = Math.max(0.06, Math.min(0.94, q.cx + rx));
          const y = Math.max(0.06, Math.min(0.94, q.cy + ry));
          dots.push({
            id: `dot-${id++}`,
            x,
            y,
            label: q.label,
            continuousTarget: y,
          });
        }
      });
      break;
    }

    case 'outliers': {
      // Linear correlation with 4-5 extreme outliers
      let id = 1;
      const baseCount = count - 5;
      const slope = 0.65;
      const intercept = 0.18;

      for (let i = 0; i < baseCount; i++) {
        const x = 0.12 + (i / baseCount) * 0.76 + (Math.random() - 0.5) * 0.04;
        const y = slope * x + intercept + (Math.random() - 0.5) * 0.06;
        const clampedY = Math.max(0.08, Math.min(0.92, y));
        dots.push({
          id: `dot-${id++}`,
          x: Math.max(0.06, Math.min(0.94, x)),
          y: clampedY,
          label: clampedY > 0.5 ? 1 : 0,
          continuousTarget: clampedY,
        });
      }

      // 5 severe adversarial outliers
      const outliers = [
        { x: 0.18, y: 0.88, label: 1 },
        { x: 0.25, y: 0.82, label: 1 },
        { x: 0.82, y: 0.15, label: 0 },
        { x: 0.88, y: 0.22, label: 0 },
        { x: 0.50, y: 0.90, label: 1 },
      ];
      outliers.forEach((o) => {
        dots.push({
          id: `dot-outlier-${id++}`,
          x: o.x,
          y: o.y,
          label: o.label,
          continuousTarget: o.y,
        });
      });
      break;
    }
  }

  return dots;
}
