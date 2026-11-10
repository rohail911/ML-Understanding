import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, ArrowRight, ArrowLeft, RotateCcw, CheckCircle, Info } from 'lucide-react';

interface JourneyChapter {
  id: number;
  title: string;
  subtitle: string;
  concept: string;
  description: string;
  takeaway: string;
}

const CHAPTERS: JourneyChapter[] = [
  {
    id: 1,
    title: 'The Birth of a Data Dot',
    subtitle: 'Features & Coordinate Space',
    concept: 'Coordinates as Measurements',
    description:
      'In Machine Learning, every observation (a customer, a medical scan, a house) is represented as a solitary dot in multi-dimensional space. Here in 2D space, our dot "Pip" has two features: X₁ (e.g., Size) and X₂ (e.g., Weight).',
    takeaway: 'Features define a dot’s location. Distance between dots measures their similarity.',
  },
  {
    id: 2,
    title: 'Finding Your Dot Family',
    subtitle: 'K-Means Clustering',
    concept: 'Unsupervised Attraction',
    description:
      'Pip is unlabelled and lonely. Suddenly, K-Means deploys K centroid anchors. Each data dot calculates its Euclidean distance to all centroids. The closest centroid adopts Pip, and in return, Pip pulls the centroid closer to the cluster center!',
    takeaway: 'Unsupervised learning discovers natural geometric clusters without human labels.',
  },
  {
    id: 3,
    title: 'Democratic Peer Review',
    subtitle: 'K-Nearest Neighbors (KNN)',
    concept: 'Instance-Based Voting',
    description:
      'Pip arrives in an unfamiliar neighborhood where some dots are Orange and some are Cyan. Pip casts a radar beam outward until it encounters its K closest neighbors. The neighbors hold an election to decide Pip’s true class.',
    takeaway: 'KNN makes zero advance assumptions; it evaluates decisions purely on local consensus.',
  },
  {
    id: 4,
    title: 'Walking Down the Loss Valley',
    subtitle: 'Gradient Descent & Linear Fitting',
    concept: 'Optimization & Residuals',
    description:
      'Pip and its sibling dots want a single line that describes their shared trend. The algorithm calculates the residual gap (error stick) between Pip and the line. By taking small steps opposite the gradient, the parameters roll down the parabolic loss bowl to optimal alignment.',
    takeaway: 'Gradient descent minimizes prediction error step-by-step through calculus.',
  },
  {
    id: 5,
    title: 'Standing Guard at the Frontier',
    subtitle: 'Support Vector Machine (SVM)',
    concept: 'Maximum Margin Hyperplane',
    description:
      'When two opposing armies of dots face each other, SVM does not just draw any dividing line—it looks for the widest possible safety highway. Pip happens to sit right on the boundary edge, transforming Pip into a critical Support Vector!',
    takeaway: 'Only boundary dots dictate the hyperplane; interior dots have zero influence.',
  },
  {
    id: 6,
    title: 'Synaptic Transformation',
    subtitle: 'Deep Neural Networks',
    concept: 'Non-Linear Representation',
    description:
      'Pip enters a multi-layer neural network. Its coordinates are multiplied by synaptic weights, summed, and pushed through non-linear activation gates (ReLU / Sigmoid). Layer by layer, the space twists until complex spirals can be cleanly untangled.',
    takeaway: 'Deep layers warp feature space so complex non-linear patterns become linearly separable.',
  },
];

export const DotJourney: React.FC = () => {
  const [currentChapterIdx, setCurrentChapterIdx] = useState<number>(0);
  const chapter = CHAPTERS[currentChapterIdx];
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animRef = useRef<number | null>(null);

  // Chapter 1-6 canvas simulation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const render = () => {
      time += 0.02;
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Coordinate axes
      ctx.strokeStyle = 'rgba(71, 85, 105, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(40, h - 40);
      ctx.lineTo(w - 40, h - 40);
      ctx.moveTo(40, h - 40);
      ctx.lineTo(40, 40);
      ctx.stroke();

      // Axis labels
      ctx.font = '500 11px "Plus Jakarta Sans", sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.fillText('Feature X₁ (e.g. Size)', w - 140, h - 20);
      ctx.fillText('Feature X₂ (e.g. Weight)', 50, 30);

      // Chapter-specific animations
      if (chapter.id === 1) {
        // Pip the Dot with pulsating coordinate callouts
        const pipX = cx + Math.sin(time) * 40;
        const pipY = cy + Math.cos(time) * 30;

        // Projection dashed lines to axes
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pipX, pipY);
        ctx.lineTo(pipX, h - 40);
        ctx.moveTo(pipX, pipY);
        ctx.lineTo(40, pipY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Axis coordinate markers
        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.arc(pipX, h - 40, 4, 0, Math.PI * 2);
        ctx.arc(40, pipY, 4, 0, Math.PI * 2);
        ctx.fill();

        // Pip
        ctx.fillStyle = '#06b6d4';
        ctx.shadowColor = '#06b6d4';
        ctx.shadowBlur = 15;
        ctx.beginPath();
        ctx.arc(pipX, pipY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Label
        ctx.font = '600 13px "Plus Jakarta Sans", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Pip: (X₁ = 0.62, X₂ = 0.54)', pipX + 16, pipY - 12);
      } else if (chapter.id === 2) {
        // K-Means: 2 Centroids attracting dots
        const c1x = cx - 120;
        const c1y = cy - 20;
        const c2x = cx + 120;
        const c2y = cy + 20;

        // Centroid 1
        ctx.fillStyle = '#06b6d4';
        ctx.beginPath();
        ctx.arc(c1x, c1y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#06b6d4';
        ctx.strokeRect(c1x - 10, c1y - 10, 20, 20);

        // Centroid 2
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(c2x, c2y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#f59e0b';
        ctx.strokeRect(c2x - 10, c2y - 10, 20, 20);

        // Surrounding dots with gravity tether
        for (let i = 0; i < 16; i++) {
          const isLeft = i < 8;
          const targetX = isLeft ? c1x : c2x;
          const targetY = isLeft ? c1y : c2y;
          const angle = (i % 8) * (Math.PI / 4) + time * 0.4;
          const r = 45 + Math.sin(time + i) * 10;
          const dx = targetX + Math.cos(angle) * r;
          const dy = targetY + Math.sin(angle) * r;

          ctx.strokeStyle = isLeft ? 'rgba(6, 182, 212, 0.25)' : 'rgba(245, 158, 11, 0.25)';
          ctx.beginPath();
          ctx.moveTo(targetX, targetY);
          ctx.lineTo(dx, dy);
          ctx.stroke();

          ctx.fillStyle = isLeft ? '#06b6d4' : '#f59e0b';
          ctx.beginPath();
          ctx.arc(dx, dy, i === 3 ? 7 : 4, 0, Math.PI * 2);
          ctx.fill();

          if (i === 3) {
            ctx.fillStyle = '#ffffff';
            ctx.font = '600 11px "Plus Jakarta Sans"';
            ctx.fillText('Pip (Adopted!)', dx + 12, dy - 6);
          }
        }
      } else if (chapter.id === 3) {
        // KNN: Query dot with expanding radar beam
        const pipX = cx;
        const pipY = cy;
        const radarR = 70 + ((time * 30) % 90);

        ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pipX, pipY, radarR, 0, Math.PI * 2);
        ctx.stroke();

        // Pip
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#ffffff';
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(pipX, pipY, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillText('Pip (Querying)', pipX + 12, pipY - 8);

        // Surrounding voters
        const voters = [
          { x: cx - 50, y: cy - 40, label: 0 },
          { x: cx - 40, y: cy + 50, label: 0 },
          { x: cx + 45, y: cy - 35, label: 1 },
          { x: cx + 60, y: cy + 30, label: 0 },
          { x: cx + 90, y: cy - 60, label: 1 },
        ];

        voters.forEach((v) => {
          const d = Math.hypot(v.x - pipX, v.y - pipY);
          if (d <= radarR) {
            ctx.strokeStyle = v.label === 0 ? '#06b6d4' : '#f59e0b';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(pipX, pipY);
            ctx.lineTo(v.x, v.y);
            ctx.stroke();
          }

          ctx.fillStyle = v.label === 0 ? '#06b6d4' : '#f59e0b';
          ctx.beginPath();
          ctx.arc(v.x, v.y, 5, 0, Math.PI * 2);
          ctx.fill();
        });
      } else if (chapter.id === 4) {
        // Linear Regression: Line fitting with residual sticks
        const slope = 0.5 + Math.sin(time * 0.7) * 0.15;
        const intercept = h * 0.7;

        // Line
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(40, intercept - slope * 40);
        ctx.lineTo(w - 40, intercept - slope * (w - 40));
        ctx.stroke();

        // Dots & residual sticks
        for (let i = 0; i < 12; i++) {
          const dx = 80 + i * (w - 160) / 11;
          const trueY = intercept - slope * dx + (Math.sin(i * 1.5) * 35);
          const lineY = intercept - slope * dx;

          // Residual stick
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.5)';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(dx, trueY);
          ctx.lineTo(dx, lineY);
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(dx, trueY, i === 5 ? 7 : 4, 0, Math.PI * 2);
          ctx.fill();

          if (i === 5) {
            ctx.fillStyle = '#ffffff';
            ctx.fillText('Pip: Residual e = -14px', dx + 12, trueY - 8);
          }
        }
      } else if (chapter.id === 5) {
        // SVM: Hyperplane with margin and support vectors
        const hx1 = cx - 140;
        const hy1 = cy + 100;
        const hx2 = cx + 140;
        const hy2 = cy - 100;

        // Hyperplane
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(hx1, hy1);
        ctx.lineTo(hx2, hy2);
        ctx.stroke();

        // Margins
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(hx1 - 30, hy1 - 20);
        ctx.lineTo(hx2 - 30, hy2 - 20);
        ctx.moveTo(hx1 + 30, hy1 + 20);
        ctx.lineTo(hx2 + 30, hy2 + 20);
        ctx.stroke();
        ctx.setLineDash([]);

        // Pip as Support Vector
        const svX = cx - 30;
        const svY = cy - 20;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(svX, svY, 9, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(svX, svY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Pip (Support Vector)', svX + 14, svY);
      } else if (chapter.id === 6) {
        // Neural Net: Synaptic pulses
        const layers = [
          [cy - 60, cy, cy + 60],
          [cy - 80, cy - 25, cy + 25, cy + 80],
          [cy],
        ];
        const layerX = [cx - 140, cx, cx + 140];

        // Synapses
        layerX.forEach((lx, li) => {
          if (li < layerX.length - 1) {
            const nextX = layerX[li + 1];
            layers[li].forEach((y1) => {
              layers[li + 1].forEach((y2) => {
                ctx.strokeStyle = 'rgba(236, 72, 153, 0.15)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(lx, y1);
                ctx.lineTo(nextX, y2);
                ctx.stroke();

                // Travelling signal pulse
                const progress = (time * 1.5 + (y1 + y2) * 0.01) % 1;
                const px = lx + (nextX - lx) * progress;
                const py = y1 + (y2 - y1) * progress;
                ctx.fillStyle = '#ec4899';
                ctx.beginPath();
                ctx.arc(px, py, 2, 0, Math.PI * 2);
                ctx.fill();
              });
            });
          }
        });

        // Layer nodes
        layerX.forEach((lx, li) => {
          layers[li].forEach((ly, ni) => {
            ctx.fillStyle = li === 0 && ni === 1 ? '#06b6d4' : '#ffffff';
            ctx.shadowColor = '#ec4899';
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.arc(lx, ly, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
          });
        });
      }

      animRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [chapter.id]);

  return (
    <div className="flex-1 flex flex-col md:flex-row h-[calc(100vh-61px)] bg-slate-950 overflow-hidden">
      {/* Visual Simulation Stage (Left) */}
      <div className="flex-1 relative bg-slate-950 flex items-center justify-center p-6 border-b md:border-b-0 md:border-r border-slate-800">
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="w-full max-w-2xl h-auto aspect-4/3 rounded-xl border border-slate-800 bg-slate-900/60 shadow-2xl block"
        />

        {/* Chapter Indicator Strip */}
        <div className="absolute top-6 left-6 flex items-center gap-2">
          {CHAPTERS.map((ch, idx) => (
            <button
              key={ch.id}
              onClick={() => setCurrentChapterIdx(idx)}
              className={`h-2 rounded-full transition-all ${
                idx === currentChapterIdx
                  ? 'w-8 bg-cyan-400'
                  : idx < currentChapterIdx
                  ? 'w-3 bg-cyan-800'
                  : 'w-3 bg-slate-800'
              }`}
              title={ch.title}
            />
          ))}
        </div>
      </div>

      {/* Story Narrative Deck (Right) */}
      <div className="w-full md:w-[460px] bg-slate-900/90 border-slate-800 p-8 flex flex-col justify-between shrink-0 overflow-y-auto">
        <div className="space-y-6">
          <div>
            <div className="flex items-center gap-2 text-xs text-cyan-400 font-semibold mb-2 tracking-wider uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              <span>
                Chapter {chapter.id} of {CHAPTERS.length}
              </span>
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">{chapter.title}</h2>
            <div className="text-sm font-medium text-slate-400 mt-1">{chapter.subtitle}</div>
          </div>

          <div className="p-4 bg-slate-950/70 border border-slate-800 rounded-xl space-y-2">
            <div className="text-xs font-semibold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" />
              <span>Core Principle: {chapter.concept}</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">{chapter.description}</p>
          </div>

          <div className="p-4 bg-emerald-950/20 border border-emerald-800/40 rounded-xl text-xs text-emerald-200">
            <div className="font-semibold text-emerald-300 mb-1 flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
              <span>Key Intuition:</span>
            </div>
            <p className="leading-relaxed">{chapter.takeaway}</p>
          </div>
        </div>

        {/* Navigation Step Buttons */}
        <div className="pt-6 border-t border-slate-800 flex items-center justify-between gap-4 mt-6">
          <button
            onClick={() => setCurrentChapterIdx((i) => Math.max(0, i - 1))}
            disabled={currentChapterIdx === 0}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border transition-colors ${
              currentChapterIdx === 0
                ? 'opacity-40 cursor-not-allowed border-slate-800 text-slate-600'
                : 'border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800'
            }`}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Previous</span>
          </button>

          <span className="text-xs font-mono text-slate-500">
            {currentChapterIdx + 1} / {CHAPTERS.length}
          </span>

          <button
            onClick={() =>
              setCurrentChapterIdx((i) => (i < CHAPTERS.length - 1 ? i + 1 : 0))
            }
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition-colors shadow-sm"
          >
            <span>{currentChapterIdx === CHAPTERS.length - 1 ? 'Restart Journey' : 'Next Chapter'}</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
