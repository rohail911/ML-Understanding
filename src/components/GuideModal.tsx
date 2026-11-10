import React from 'react';
import { X, MousePointer, Sliders, Layers, Sparkles } from 'lucide-react';
import { ALGORITHMS } from '../utils/algorithms';

interface GuideModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const GuideModal: React.FC<GuideModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h3 className="text-lg font-bold text-white tracking-tight">
              DotML Exploration Guide
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              How the particle universe visualizes machine learning mathematics
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-6 overflow-y-auto text-xs text-slate-300">
          {/* Quick Start Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1.5">
              <div className="flex items-center gap-2 font-semibold text-cyan-400 text-sm">
                <MousePointer className="w-4 h-4" />
                <span>Live Dot Manipulation</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Click any dot to open its mathematical inspector. Drag dots around to watch how
                decision boundaries, centroids, and regression slopes adapt dynamically.
              </p>
            </div>

            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1.5">
              <div className="flex items-center gap-2 font-semibold text-amber-400 text-sm">
                <Sliders className="w-4 h-4" />
                <span>Real-Time Parameters</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Tune hyperparameters like K, Learning Rate (α), and Regularization (C). The canvas
                instantly re-renders the mathematical consequences.
              </p>
            </div>

            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1.5">
              <div className="flex items-center gap-2 font-semibold text-violet-400 text-sm">
                <Layers className="w-4 h-4" />
                <span>Dual Model Arena</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Compare two models side-by-side on identical dots. Notice why linear SVM fails on
                intertwined moons where KNN or neural nets succeed!
              </p>
            </div>

            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1.5">
              <div className="flex items-center gap-2 font-semibold text-emerald-400 text-sm">
                <Sparkles className="w-4 h-4" />
                <span>The Dot Journey</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Follow an unlabelled data dot ("Pip") through 6 chapters explaining the fundamental
                ideas of modern AI from first principles.
              </p>
            </div>

            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-1.5 sm:col-span-2">
              <div className="flex items-center gap-2 font-semibold text-teal-400 text-sm">
                <span>Universal Data Ingestion (Any Format)</span>
              </div>
              <p className="text-slate-400 leading-relaxed">
                Bring your own real-world data: upload CSV/JSON/TSV files, paste Python arrays or spreadsheet
                rows, or paste messy unstructured text to have Gemini 3.8 Flash automatically normalize
                and structure it into interactive dots. Any of the 12 algorithms can be immediately applied!
              </p>
            </div>
          </div>

          {/* Algorithm Legend */}
          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">
              Algorithms in the Cosmos
            </h4>
            <div className="space-y-2">
              {ALGORITHMS.map((algo) => (
                <div
                  key={algo.id}
                  className="flex items-start gap-3 p-2.5 rounded-lg bg-slate-950/50 border border-slate-800/80"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full mt-1 shrink-0"
                    style={{ backgroundColor: algo.color }}
                  />
                  <div>
                    <div className="font-semibold text-slate-200">{algo.name}</div>
                    <div className="text-[11px] text-slate-400">{algo.tagline}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-950 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            Got it, Let's Explore
          </button>
        </div>
      </div>
    </div>
  );
};
