import React, { useEffect, useRef } from 'react';
import { AnimatedCounter } from './AnimatedCounter';
import { AlgorithmId } from '../types/ml';
import { Activity, TrendingDown, CheckCircle2, AlertTriangle, Zap, Flame } from 'lucide-react';

export interface PerformanceMetrics {
  primaryLabel: string;
  primaryValue: number;
  primaryDecimals?: number;
  primaryPrefix?: string;
  primarySuffix?: string;

  secondaryLabel: string;
  secondaryValue: number;
  secondaryDecimals?: number;
  secondaryPrefix?: string;
  secondarySuffix?: string;

  tertiaryLabel: string;
  tertiaryValue: number;
  tertiaryDecimals?: number;
  tertiaryPrefix?: string;
  tertiarySuffix?: string;

  healthStatus: 'optimal' | 'training' | 'warning' | 'idle';
  statusText: string;
  history: number[]; // e.g. loss or accuracy history
}

interface MetricsDashboardProps {
  algorithmId: AlgorithmId;
  metrics: PerformanceMetrics;
}

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  algorithmId,
  metrics,
}) => {
  const sparklineRef = useRef<HTMLCanvasElement | null>(null);

  // Draw real-time sparkline history
  useEffect(() => {
    const canvas = sparklineRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const history = metrics.history;
    if (!history || history.length < 2) {
      // Flat line fallback
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      return;
    }

    const min = Math.min(...history);
    const max = Math.max(...history);
    const range = max - min || 0.0001;

    // Draw area fill
    ctx.beginPath();
    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * w;
      const normalized = (val - min) / range;
      const y = h - 6 - normalized * (h - 12);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Close path for gradient fill under the curve
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
    fillGrad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
    fillGrad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Draw line
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((val, i) => {
      const x = (i / (history.length - 1)) * w;
      const normalized = (val - min) / range;
      const y = h - 6 - normalized * (h - 12);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Current point dot
    const lastX = w;
    const lastNorm = (history[history.length - 1] - min) / range;
    const lastY = h - 6 - lastNorm * (h - 12);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#06b6d4';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(lastX - 2, lastY, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }, [metrics.history]);

  return (
    <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-4 backdrop-blur shadow-inner">
      {/* Header & Status Indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold text-white uppercase tracking-wider">
            Real-Time Telemetry Dashboard
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] font-medium">
          {metrics.healthStatus === 'optimal' && (
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{metrics.statusText}</span>
            </span>
          )}
          {metrics.healthStatus === 'training' && (
            <span className="text-cyan-400 flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 animate-pulse" />
              <span>{metrics.statusText}</span>
            </span>
          )}
          {metrics.healthStatus === 'warning' && (
            <span className="text-amber-400 flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{metrics.statusText}</span>
            </span>
          )}
          {metrics.healthStatus === 'idle' && (
            <span className="text-slate-400 flex items-center gap-1">
              <span>{metrics.statusText}</span>
            </span>
          )}
        </div>
      </div>

      {/* 3 Animated Metric Tiles */}
      <div className="grid grid-cols-3 gap-2.5">
        {/* Primary Metric Tile (e.g. Loss or Inertia) */}
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg flex flex-col justify-between">
          <span className="text-[10px] font-medium text-slate-400 uppercase tracking-tight">
            {metrics.primaryLabel}
          </span>
          <div className="mt-1">
            <AnimatedCounter
              value={metrics.primaryValue}
              decimals={metrics.primaryDecimals ?? 4}
              prefix={metrics.primaryPrefix ?? ''}
              suffix={metrics.primarySuffix ?? ''}
              className="text-base font-bold text-white"
            />
          </div>
        </div>

        {/* Secondary Metric Tile (e.g. Accuracy or Margin) */}
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg flex flex-col justify-between">
          <span className="text-[10px] font-medium text-slate-400 uppercase tracking-tight">
            {metrics.secondaryLabel}
          </span>
          <div className="mt-1">
            <AnimatedCounter
              value={metrics.secondaryValue}
              decimals={metrics.secondaryDecimals ?? 1}
              prefix={metrics.secondaryPrefix ?? ''}
              suffix={metrics.secondarySuffix ?? ''}
              className="text-base font-bold text-cyan-400"
            />
          </div>
        </div>

        {/* Tertiary Metric Tile (e.g. Step, Clusters, Support Vectors) */}
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-lg flex flex-col justify-between">
          <span className="text-[10px] font-medium text-slate-400 uppercase tracking-tight">
            {metrics.tertiaryLabel}
          </span>
          <div className="mt-1">
            <AnimatedCounter
              value={metrics.tertiaryValue}
              decimals={metrics.tertiaryDecimals ?? 0}
              prefix={metrics.tertiaryPrefix ?? ''}
              suffix={metrics.tertiarySuffix ?? ''}
              className="text-base font-bold text-slate-300"
            />
          </div>
        </div>
      </div>

      {/* Real-Time Sparkline Trend */}
      <div>
        <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5 font-medium">
          <span className="flex items-center gap-1 text-slate-300">
            <TrendingDown className="w-3.5 h-3.5 text-cyan-400" />
            <span>Loss / Performance Trajectory</span>
          </span>
          <span className="font-mono text-[10px] text-slate-500">
            {metrics.history.length > 0 ? `${metrics.history.length} updates` : 'Standing by'}
          </span>
        </div>
        <canvas
          ref={sparklineRef}
          width={320}
          height={48}
          className="w-full h-12 block bg-slate-950/80 rounded border border-slate-800/80"
        />
      </div>
    </div>
  );
};
