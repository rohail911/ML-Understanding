import React, { useEffect, useRef, useState } from 'react';
import { ALGORITHMS } from '../utils/algorithms';
import { AlgorithmId, AlgorithmMetadata } from '../types/ml';
import { Sparkles, ArrowRight, Play, Maximize2, RotateCcw, Filter } from 'lucide-react';

interface CosmosWorldProps {
  onSelectAlgorithm: (id: AlgorithmId) => void;
}

interface ConstellationNode {
  algo: AlgorithmMetadata;
  x: number;
  y: number;
  radius: number;
  particles: Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    targetX?: number;
    targetY?: number;
    color: string;
    size: number;
    group?: number;
    angle?: number;
    orbitR?: number;
  }>;
}

export const CosmosWorld: React.FC<CosmosWorldProps> = ({ onSelectAlgorithm }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoveredAlgo, setHoveredAlgo] = useState<AlgorithmMetadata | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const animFrameRef = useRef<number | null>(null);

  // Mouse coordinates relative to canvas center
  const mouseRef = useRef<{ x: number; y: number; isDown: boolean; startX: number; startY: number }>({
    x: -9999,
    y: -9999,
    isDown: false,
    startX: 0,
    startY: 0,
  });

  // Pan offset
  const panRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Constellations layout in cosmos
  const constellationsRef = useRef<ConstellationNode[]>([]);

  // Initialize constellations and their orbiting algorithm dots
  useEffect(() => {
    const nodes: ConstellationNode[] = [];
    
    // Group into inner core (fundamental models) and outer orbital shell
    const innerRingIds: AlgorithmId[] = ['linear-regression', 'logistic-regression', 'kmeans', 'pca'];
    const outerRingAlgos = ALGORITHMS.filter((a) => !innerRingIds.includes(a.id));
    const innerRingAlgos = innerRingIds.map((id) => ALGORITHMS.find((a) => a.id === id)!).filter(Boolean);

    // Inner ring (radius 220)
    innerRingAlgos.forEach((algo, idx) => {
      const angle = (idx / innerRingAlgos.length) * Math.PI * 2 - Math.PI / 4;
      const dist = 220;
      const cx = Math.cos(angle) * dist;
      const cy = Math.sin(angle) * dist;
      nodes.push(createConstellationNode(algo, cx, cy));
    });

    // Outer ring (radius 430)
    outerRingAlgos.forEach((algo, idx) => {
      const angle = (idx / outerRingAlgos.length) * Math.PI * 2 - Math.PI / 2;
      const dist = 430;
      const cx = Math.cos(angle) * dist;
      const cy = Math.sin(angle) * dist;
      nodes.push(createConstellationNode(algo, cx, cy));
    });

    function createConstellationNode(algo: AlgorithmMetadata, cx: number, cy: number): ConstellationNode {
      const particles: ConstellationNode['particles'] = [];
      const particleCount = 22;

      for (let i = 0; i < particleCount; i++) {
        const pAngle = (i / particleCount) * Math.PI * 2;
        const orbitR = 30 + Math.random() * 45;
        let color = algo.color;
        let group = 0;

        // Custom styling for algorithm-specific dots
        if (
          algo.id === 'svm' ||
          algo.id === 'knn' ||
          algo.id === 'decision-tree' ||
          algo.id === 'logistic-regression' ||
          algo.id === 'naive-bayes' ||
          algo.id === 'random-forest'
        ) {
          group = i % 2;
          color = group === 0 ? '#06b6d4' : '#f59e0b';
        } else if (algo.id === 'kmeans' || algo.id === 'hierarchical') {
          group = i % 3;
          const colors = ['#06b6d4', '#f59e0b', '#8b5cf6'];
          color = colors[group];
        } else if (algo.id === 'dbscan') {
          if (i === 0 || i === 7) {
            color = '#f43f5e'; // Noise outliers
          } else {
            color = i % 2 === 0 ? '#10b981' : '#06b6d4';
          }
        }

        particles.push({
          x: cx + Math.cos(pAngle) * orbitR,
          y: cy + Math.sin(pAngle) * orbitR,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          color,
          size: 2.5 + Math.random() * 2.5,
          group,
          angle: pAngle,
          orbitR,
        });
      }

      return {
        algo,
        x: cx,
        y: cy,
        radius: 75,
        particles,
      };
    }

    constellationsRef.current = nodes;
  }, []);

  // Animation render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const render = () => {
      time += 0.015;
      const width = canvas.width;
      const height = canvas.height;
      const halfW = width / 2;
      const halfH = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Deep celestial cosmic backdrop
      ctx.save();
      ctx.translate(halfW + panRef.current.x, halfH + panRef.current.y);
      ctx.scale(zoomLevel, zoomLevel);

      // 1. Draw cosmic grid & orbit rings
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)';
      ctx.lineWidth = 1;

      // Concentric orbital guidance lines
      [150, 320, 480].forEach((r) => {
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.stroke();
      });

      // Subtle coordinate crosshair
      ctx.beginPath();
      ctx.moveTo(-600, 0);
      ctx.lineTo(600, 0);
      ctx.moveTo(0, -600);
      ctx.lineTo(0, -600);
      ctx.stroke();

      // 2. Inter-constellation neural energy filaments
      constellationsRef.current.forEach((c1, i) => {
        constellationsRef.current.forEach((c2, j) => {
          if (j > i) {
            const d = Math.hypot(c1.x - c2.x, c1.y - c2.y);
            if (d < 450) {
              const alpha = Math.max(0.03, (1 - d / 450) * 0.15);
              ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(c1.x, c1.y);
              ctx.lineTo(c2.x, c2.y);
              ctx.stroke();
            }
          }
        });
      });

      // 3. Render each Algorithm Constellation
      const mouseWorldX = (mouseRef.current.x - halfW - panRef.current.x) / zoomLevel;
      const mouseWorldY = (mouseRef.current.y - halfH - panRef.current.y) / zoomLevel;

      let foundHover: AlgorithmMetadata | null = null;

      constellationsRef.current.forEach((c) => {
        const isDimmed =
          selectedCategory !== 'All' && c.algo.category !== selectedCategory;
        const distToMouse = Math.hypot(c.x - mouseWorldX, c.y - mouseWorldY);
        const isHovered = distToMouse < c.radius + 15;

        if (isHovered && !foundHover) {
          foundHover = c.algo;
        }

        ctx.save();
        ctx.globalAlpha = isDimmed ? 0.25 : 1;

        // Outer glow aura
        const glowRad = isHovered ? c.radius * 1.4 : c.radius;
        const grad = ctx.createRadialGradient(c.x, c.y, 10, c.x, c.y, glowRad);
        grad.addColorStop(0, `rgba(${c.algo.accentRgb}, ${isHovered ? 0.35 : 0.15})`);
        grad.addColorStop(0.7, `rgba(${c.algo.accentRgb}, ${isHovered ? 0.12 : 0.04})`);
        grad.addColorStop(1, 'transparent');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(c.x, c.y, glowRad, 0, Math.PI * 2);
        ctx.fill();

        // Constellation perimeter ring
        ctx.strokeStyle = `rgba(${c.algo.accentRgb}, ${isHovered ? 0.7 : 0.25})`;
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.setLineDash([4, 6]);
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Algorithm Signature Visual Animations within the constellation
        if (c.algo.id === 'linear-regression') {
          // Animated fitted line
          const lineAngle = Math.sin(time * 0.8) * 0.3 - 0.4;
          const lx = Math.cos(lineAngle) * 45;
          const ly = Math.sin(lineAngle) * 45;
          ctx.strokeStyle = `rgba(59, 130, 246, ${isHovered ? 0.9 : 0.6})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(c.x - lx, c.y - ly);
          ctx.lineTo(c.x + lx, c.y + ly);
          ctx.stroke();
        } else if (c.algo.id === 'svm') {
          // Separating Hyperplane and margins
          ctx.strokeStyle = `rgba(139, 92, 246, ${isHovered ? 0.9 : 0.6})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(c.x - 45, c.y + 35);
          ctx.lineTo(c.x + 45, c.y - 35);
          ctx.stroke();

          // Dotted margins
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(c.x - 45, c.y + 20);
          ctx.lineTo(c.x + 45, c.y - 50);
          ctx.moveTo(c.x - 45, c.y + 50);
          ctx.lineTo(c.x + 45, c.y - 20);
          ctx.stroke();
          ctx.setLineDash([]);
        } else if (c.algo.id === 'kmeans') {
          // 3 Centroid stars pulsing
          for (let k = 0; k < 3; k++) {
            const ca = (k / 3) * Math.PI * 2 + time * 0.5;
            const kx = c.x + Math.cos(ca) * 22;
            const ky = c.y + Math.sin(ca) * 22;
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(kx, ky, 3.5, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(c.x, c.y);
            ctx.lineTo(kx, ky);
            ctx.stroke();
          }
        } else if (c.algo.id === 'knn') {
          // Pulsing radar sweep
          const sweepR = 20 + ((time * 25) % 45);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(c.x, c.y, sweepR, 0, Math.PI * 2);
          ctx.stroke();
        } else if (c.algo.id === 'decision-tree') {
          // Axis-aligned orthogonal splits
          ctx.strokeStyle = 'rgba(249, 115, 22, 0.5)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(c.x, c.y - 45);
          ctx.lineTo(c.x, c.y + 45);
          ctx.moveTo(c.x, c.y);
          ctx.lineTo(c.x + 45, c.y);
          ctx.stroke();
        } else if (c.algo.id === 'pca') {
          // First principal eigenvector with projection shadows
          const pcaAngle = Math.PI / 4;
          const px = Math.cos(pcaAngle) * 45;
          const py = Math.sin(pcaAngle) * 45;
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(c.x - px, c.y - py);
          ctx.lineTo(c.x + px, c.y + py);
          ctx.stroke();
        } else if (c.algo.id === 'neural-net') {
          // Synaptic threads connecting layers
          const layers = [-25, 0, 25];
          layers.forEach((lx1) => {
            layers.forEach((lx2) => {
              if (lx2 > lx1) {
                ctx.strokeStyle = 'rgba(236, 72, 153, 0.18)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(c.x + lx1, c.y - 15);
                ctx.lineTo(c.x + lx2, c.y + 15);
                ctx.stroke();
              }
            });
          });
        } else if (c.algo.id === 'logistic-regression') {
          // Smooth sigmoid curve
          ctx.strokeStyle = `rgba(6, 182, 212, ${isHovered ? 0.9 : 0.6})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          for (let sx = -40; sx <= 40; sx += 4) {
            const z = (sx / 40) * 4;
            const sig = 1 / (1 + Math.exp(-z));
            const sy = (0.5 - sig) * 40;
            if (sx === -40) ctx.moveTo(c.x + sx, c.y + sy);
            else ctx.lineTo(c.x + sx, c.y + sy);
          }
          ctx.stroke();
        } else if (c.algo.id === 'dbscan') {
          // Expanding density bubble wave
          const waveR = 15 + ((time * 20) % 35);
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(c.x, c.y, waveR, 0, Math.PI * 2);
          ctx.stroke();

          // Outlier red speck
          const ox = c.x + Math.cos(time * 0.4) * 48;
          const oy = c.y + Math.sin(time * 0.4) * 48;
          ctx.fillStyle = '#f43f5e';
          ctx.beginPath();
          ctx.arc(ox, oy, 2.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (c.algo.id === 'naive-bayes') {
          // Dual probability density ellipses
          ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.ellipse(c.x - 14, c.y, 22, 14, Math.PI / 6, 0, Math.PI * 2);
          ctx.stroke();

          ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
          ctx.beginPath();
          ctx.ellipse(c.x + 14, c.y, 20, 12, -Math.PI / 6, 0, Math.PI * 2);
          ctx.stroke();
        } else if (c.algo.id === 'random-forest') {
          // Branching mini ensemble tree lines
          const angles = [-0.6, 0, 0.6];
          angles.forEach((ang) => {
            const tx = c.x + Math.sin(ang) * 35;
            const ty = c.y - Math.cos(ang) * 30;
            ctx.strokeStyle = 'rgba(217, 119, 6, 0.6)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(c.x, c.y + 10);
            ctx.lineTo(tx, ty);
            ctx.stroke();

            // Leaf nodes
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.arc(tx, ty, 2.5, 0, Math.PI * 2);
            ctx.fill();
          });
        } else if (c.algo.id === 'hierarchical') {
          // Dendrogram fork brackets
          ctx.strokeStyle = 'rgba(20, 184, 166, 0.7)';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(c.x - 25, c.y + 20);
          ctx.lineTo(c.x - 25, c.y);
          ctx.lineTo(c.x, c.y);
          ctx.lineTo(c.x, c.y - 18);
          ctx.lineTo(c.x, c.y);
          ctx.lineTo(c.x + 25, c.y);
          ctx.lineTo(c.x + 25, c.y + 20);
          ctx.stroke();
        }

        // Orbiting particles (The Living Data Dots)
        c.particles.forEach((p) => {
          // Update orbit position
          if (p.angle !== undefined && p.orbitR !== undefined) {
            p.angle += 0.012 + (p.size / 10) * 0.005;
            p.x = c.x + Math.cos(p.angle) * p.orbitR;
            p.y = c.y + Math.sin(p.angle) * p.orbitR;
          }

          // Gentle mouse gravitation
          const dmp = Math.hypot(p.x - mouseWorldX, p.y - mouseWorldY);
          if (dmp < 60) {
            p.x += ((p.x - mouseWorldX) / dmp) * 1.5;
            p.y += ((p.y - mouseWorldY) / dmp) * 1.5;
          }

          // Draw the dot
          ctx.fillStyle = p.color;
          ctx.shadowColor = p.color;
          ctx.shadowBlur = isHovered ? 8 : 4;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        });

        // Center Core Node
        ctx.fillStyle = isHovered ? '#ffffff' : `rgba(${c.algo.accentRgb}, 0.9)`;
        ctx.shadowColor = `rgba(${c.algo.accentRgb}, 0.8)`;
        ctx.shadowBlur = isHovered ? 15 : 8;
        ctx.beginPath();
        ctx.arc(c.x, c.y, isHovered ? 8 : 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Label below constellation
        ctx.font = '600 13px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = isHovered ? '#ffffff' : '#cbd5e1';
        ctx.textAlign = 'center';
        ctx.fillText(c.algo.name, c.x, c.y + c.radius + 20);

        // Category subtitle
        ctx.font = '500 10px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = isHovered ? `rgba(${c.algo.accentRgb}, 1)` : '#64748b';
        ctx.fillText(c.algo.category.toUpperCase(), c.x, c.y + c.radius + 34);

        ctx.restore();
      });

      setHoveredAlgo(foundHover);
      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [selectedCategory, zoomLevel]);

  // Resize listener
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

  // Mouse interaction for pan and click
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    mouseRef.current.isDown = true;
    mouseRef.current.startX = e.clientX - panRef.current.x;
    mouseRef.current.startY = e.clientY - panRef.current.y;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    mouseRef.current.x = e.clientX - rect.left;
    mouseRef.current.y = e.clientY - rect.top;

    if (mouseRef.current.isDown) {
      panRef.current.x = e.clientX - mouseRef.current.startX;
      panRef.current.y = e.clientY - mouseRef.current.startY;
    }
  };

  const handleMouseUp = () => {
    mouseRef.current.isDown = false;
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const halfW = canvas.width / 2;
    const halfH = canvas.height / 2;

    const clickWorldX = (e.clientX - rect.left - halfW - panRef.current.x) / zoomLevel;
    const clickWorldY = (e.clientY - rect.top - halfH - panRef.current.y) / zoomLevel;

    // Check which constellation was clicked
    const clicked = constellationsRef.current.find(
      (c) => Math.hypot(c.x - clickWorldX, c.y - clickWorldY) < c.radius + 15
    );

    if (clicked) {
      onSelectAlgorithm(clicked.algo.id);
    }
  };

  const resetCosmosView = () => {
    panRef.current = { x: 0, y: 0 };
    setZoomLevel(1);
  };

  return (
    <div ref={containerRef} className="relative w-full h-[calc(100vh-61px)] overflow-hidden bg-slate-950">
      {/* Top Floating Category Filter Tabs */}
      <div className="absolute top-5 left-6 z-20 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 p-1 bg-slate-900/90 border border-slate-800 rounded-lg backdrop-blur-md shadow-lg">
          <div className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 font-medium">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            <span>Realm:</span>
          </div>
          {['All', 'Supervised', 'Unsupervised', 'Deep Learning', 'Dimensionality Reduction'].map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap ${
                selectedCategory === cat
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Floating Instructions & Zoom Controls */}
      <div className="absolute top-5 right-6 z-20 flex items-center gap-2">
        <div className="flex items-center bg-slate-900/90 border border-slate-800 rounded-lg p-1 backdrop-blur-md">
          <button
            onClick={() => setZoomLevel((z) => Math.min(1.8, z + 0.15))}
            className="px-2.5 py-1 text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors"
            title="Zoom In"
          >
            +
          </button>
          <span className="px-2 text-xs font-mono tabular-nums text-slate-400">
            {Math.round(zoomLevel * 100)}%
          </span>
          <button
            onClick={() => setZoomLevel((z) => Math.max(0.6, z - 0.15))}
            className="px-2.5 py-1 text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors"
            title="Zoom Out"
          >
            -
          </button>
          <button
            onClick={resetCosmosView}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded ml-1 transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Interactive Main Canvas */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        className="w-full h-full cursor-grab active:cursor-grabbing block"
      />

      {/* Dynamic Hover Details Card (Bottom Center / Left) */}
      {hoveredAlgo && (
        <div className="absolute bottom-6 left-6 z-20 max-w-md w-full bg-slate-900/95 border border-slate-700/80 rounded-xl p-5 backdrop-blur-lg shadow-2xl animate-in fade-in slide-in-from-bottom-2 duration-200">
          <div className="flex items-start justify-between gap-4 mb-2">
            <div>
              <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
                <span className="font-semibold text-cyan-400">{hoveredAlgo.category}</span>
                <span aria-hidden="true">·</span>
                <span>{hoveredAlgo.primaryMetric}</span>
              </div>
              <h3 className="text-lg font-bold text-white tracking-tight">{hoveredAlgo.name}</h3>
            </div>
            <button
              onClick={() => onSelectAlgorithm(hoveredAlgo.id)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-all shadow-sm shrink-0"
            >
              <span>Enter Lab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed mb-3">{hoveredAlgo.description}</p>

          <div className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800 text-xs font-mono text-cyan-300 mb-3 overflow-x-auto">
            {hoveredAlgo.mathFormula}
          </div>

          <div className="flex flex-wrap gap-1.5">
            {hoveredAlgo.keyConcepts.map((concept) => (
              <span
                key={concept}
                className="text-[11px] text-slate-400 bg-slate-800/80 border border-slate-700/60 rounded px-2 py-0.5"
              >
                {concept}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Floating Bottom Navigation Strip */}
      <div className="absolute bottom-6 right-6 z-20 hidden lg:flex items-center gap-1.5 p-1.5 bg-slate-900/90 border border-slate-800 rounded-xl backdrop-blur-md shadow-xl max-w-[62vw] overflow-x-auto">
        <span className="text-xs text-slate-400 px-2 font-medium shrink-0">Quick Teleport:</span>
        {ALGORITHMS.map((algo) => (
          <button
            key={algo.id}
            onClick={() => onSelectAlgorithm(algo.id)}
            className="px-2 py-1 text-[11px] font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-1.5 shrink-0"
          >
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: algo.color }}
            />
            <span>{algo.name.split(' ')[0]}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
