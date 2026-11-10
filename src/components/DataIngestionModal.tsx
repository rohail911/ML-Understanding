import React, { useState, useRef, useEffect } from 'react';
import {
  X,
  UploadCloud,
  FileText,
  Sparkles,
  Database,
  Sliders,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  RotateCcw,
  Code2,
  Table,
  Layers,
  Info,
} from 'lucide-react';
import { DataDot } from '../types/ml';
import {
  parseRawData,
  ParsedDataset,
  BENCHMARK_DATASETS,
  BenchmarkDataset,
} from '../utils/dataParser';

interface DataIngestionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApplyDataset: (dots: DataDot[], metadata: { name: string; description: string }) => void;
}

type TabType = 'upload' | 'paste' | 'ai' | 'benchmarks';

export const DataIngestionModal: React.FC<DataIngestionModalProps> = ({
  isOpen,
  onClose,
  onApplyDataset,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('paste');
  const [inputText, setInputText] = useState<string>('');
  const [aiInstructions, setAiInstructions] = useState<string>('');
  const [isAiLoading, setIsAiLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Parsed result state
  const [parsedData, setParsedData] = useState<ParsedDataset | null>(null);
  const [selectedF1, setSelectedF1] = useState<string>('');
  const [selectedF2, setSelectedF2] = useState<string>('');
  const [selectedLabel, setSelectedLabel] = useState<string>('');
  const [usePCA, setUsePCA] = useState<boolean>(false);

  // File upload state
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Mini preview canvas
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize with Iris benchmark on first open if empty
  useEffect(() => {
    if (isOpen && !inputText && !parsedData) {
      loadBenchmark(BENCHMARK_DATASETS[0]);
    }
  }, [isOpen]);

  // Render mini preview scatter plot
  useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !parsedData || parsedData.dots.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Dots
    parsedData.dots.forEach((dot) => {
      const cx = dot.x * width;
      const cy = (1 - dot.y) * height;
      const color = dot.label === 1 ? '#f59e0b' : '#06b6d4';

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [parsedData]);

  if (!isOpen) return null;

  // Handle parsing text
  const handleParseText = (textToParse: string, options?: { f1?: string; f2?: string; label?: string; pca?: boolean }) => {
    setErrorMsg(null);
    try {
      const res = parseRawData(textToParse, {
        feature1Col: options?.f1 || selectedF1,
        feature2Col: options?.f2 || selectedF2,
        labelCol: options?.label || selectedLabel,
        usePCA: options?.pca !== undefined ? options.pca : usePCA,
      });
      setParsedData(res);
      setSelectedF1(res.feature1);
      setSelectedF2(res.feature2);
      setSelectedLabel(res.labelColumn);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to parse data.');
      setParsedData(null);
    }
  };

  // Re-run column mapping changes
  const handleColumnMappingChange = (newF1: string, newF2: string, newLabel: string, newPCA: boolean) => {
    setSelectedF1(newF1);
    setSelectedF2(newF2);
    setSelectedLabel(newLabel);
    setUsePCA(newPCA);
    handleParseText(inputText, { f1: newF1, f2: newF2, label: newLabel, pca: newPCA });
  };

  // File drop / upload handlers
  const handleFileUpload = (file: File) => {
    setErrorMsg(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setInputText(content);
      handleParseText(content);
    };
    reader.onerror = () => {
      setErrorMsg('Failed to read file. Please try another file.');
    };
    reader.readAsText(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Load a Benchmark
  const loadBenchmark = (b: BenchmarkDataset) => {
    setInputText(b.rawText);
    handleParseText(b.rawText);
  };

  // AI Smart Ingest with Gemini
  const handleAiSmartIngest = async () => {
    if (!inputText.trim()) {
      setErrorMsg('Please paste your unstructured text, records, or logs first.');
      return;
    }
    setErrorMsg(null);
    setIsAiLoading(true);

    try {
      const response = await fetch('/api/ingest-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rawText: inputText,
          instructions: aiInstructions,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with ${response.status}`);
      }

      const resJson = await response.json();
      if (!resJson.dots || !Array.isArray(resJson.dots)) {
        throw new Error('AI could not extract valid dots from the given text.');
      }

      const dots: DataDot[] = resJson.dots.map((d: any, idx: number) => ({
        id: `custom-dot-${idx + 1}`,
        x: Math.max(0.04, Math.min(0.96, Number(d.x) || 0.5)),
        y: Math.max(0.04, Math.min(0.96, Number(d.y) || 0.5)),
        label: d.label !== undefined ? (Number(d.label) >= 1 ? 1 : 0) : 0,
        continuousTarget: d.continuousTarget !== undefined ? Number(d.continuousTarget) : Number(d.y),
      }));

      const newParsed: ParsedDataset = {
        name: resJson.datasetName || 'AI Extracted Dataset',
        description: resJson.description || 'Processed via Gemini 3.8 Flash',
        columns: [resJson.feature1Name || 'Feature 1', resJson.feature2Name || 'Feature 2', resJson.targetName || 'Class'],
        numericColumns: [resJson.feature1Name || 'Feature 1', resJson.feature2Name || 'Feature 2'],
        rawRows: resJson.dots,
        feature1: resJson.feature1Name || 'Feature 1',
        feature2: resJson.feature2Name || 'Feature 2',
        labelColumn: resJson.targetName || 'Class',
        dots,
      };

      setParsedData(newParsed);
      setSelectedF1(newParsed.feature1);
      setSelectedF2(newParsed.feature2);
      setSelectedLabel(newParsed.labelColumn);
    } catch (err: any) {
      console.error('AI Ingest Error:', err);
      setErrorMsg(err.message || 'Failed to process data with Gemini AI.');
    } finally {
      setIsAiLoading(false);
    }
  };

  // Apply and Load into Cosmos & Sandbox
  const handleApply = () => {
    if (!parsedData || parsedData.dots.length === 0) {
      setErrorMsg('Please parse or upload valid data before applying.');
      return;
    }
    onApplyDataset(parsedData.dots, {
      name: parsedData.name,
      description: parsedData.description,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-800 bg-slate-950/40">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                <span>Universal Data Ingestion</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-800/60 font-semibold">
                  Any Format → Any Algorithm
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Input your own data (CSV, JSON, TSV, Python lists, or messy text) and apply all 12 ML algorithms immediately.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 px-5 pt-3 pb-2 border-b border-slate-800 bg-slate-950/20 text-xs">
          <button
            onClick={() => setActiveTab('paste')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'paste'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>Paste Text / Code</span>
          </button>

          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'upload'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span>Upload File (.csv, .json, .tsv)</span>
          </button>

          <button
            onClick={() => setActiveTab('ai')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'ai'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>AI Smart Ingest (Unstructured)</span>
          </button>

          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeTab === 'benchmarks'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
          >
            <Table className="w-3.5 h-3.5" />
            <span>Classic Benchmark Datasets</span>
          </button>
        </div>

        {/* Modal Main Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* TAB 1: Paste Text */}
          {activeTab === 'paste' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Paste CSV, TSV (Excel copy-paste), JSON objects/arrays, or Python matrices:</span>
                <button
                  onClick={() => {
                    const sample = `age,income,credit_score,approved\n25,45000,680,1\n45,85000,740,1\n22,25000,580,0\n35,62000,710,1\n55,120000,800,1\n28,32000,610,0\n48,92000,760,1\n23,28000,590,0\n38,71000,690,1\n60,115000,790,1`;
                    setInputText(sample);
                    handleParseText(sample);
                  }}
                  className="text-cyan-400 hover:underline font-medium"
                >
                  Insert Sample CSV
                </button>
              </div>
              <textarea
                value={inputText}
                onChange={(e) => {
                  setInputText(e.target.value);
                  handleParseText(e.target.value);
                }}
                placeholder="e.g.&#10;x1, x2, label&#10;1.2, 4.5, 0&#10;3.4, 2.1, 1&#10;5.6, 7.8, 1"
                className="w-full h-36 bg-slate-950 font-mono text-xs text-slate-200 p-3 rounded-xl border border-slate-800 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 leading-relaxed resize-y"
              />
            </div>
          )}

          {/* TAB 2: Upload File */}
          {activeTab === 'upload' && (
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                dragActive
                  ? 'border-cyan-400 bg-cyan-950/20'
                  : 'border-slate-800 hover:border-slate-700 bg-slate-950/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.json,.txt"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
                className="hidden"
              />
              <UploadCloud className="w-10 h-10 text-cyan-400 mx-auto mb-3" />
              <h4 className="text-sm font-semibold text-white mb-1">
                Drag and drop your dataset file here
              </h4>
              <p className="text-xs text-slate-400 mb-2">
                Supports CSV, TSV (tab-separated), JSON (arrays or object rows), and plain text
              </p>
              <span className="inline-block px-3 py-1 text-xs font-semibold text-cyan-300 bg-cyan-950/60 border border-cyan-800/60 rounded-lg">
                Browse File
              </span>
            </div>
          )}

          {/* TAB 3: AI Smart Ingest */}
          {activeTab === 'ai' && (
            <div className="space-y-3">
              <div className="p-3 bg-amber-950/20 border border-amber-800/40 rounded-xl flex items-start gap-2.5 text-xs text-amber-200">
                <Sparkles className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <p className="leading-snug">
                  <strong>AI Ingestion Engine:</strong> Paste unstructured text, clinical patient notes,
                  chat transcripts, raw sensor telemetry, or natural language descriptions. Gemini 3.8
                  Flash will automatically extract variables, normalize coordinates to [0, 1], and assign
                  classes!
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Raw Unstructured Data / Text:
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste anything here! Example:
Patient A: age 45, cholesterol 240, had event; Patient B: age 28, cholesterol 160, no event; Patient C: age 62, cholesterol 290, had event..."
                  className="w-full h-28 bg-slate-950 font-sans text-xs text-slate-200 p-3 rounded-xl border border-slate-800 focus:outline-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 leading-relaxed resize-y"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Optional Guidance for Gemini (e.g., "focus on age vs spending"):
                </label>
                <input
                  type="text"
                  value={aiInstructions}
                  onChange={(e) => setAiInstructions(e.target.value)}
                  placeholder="e.g. Map X to temperature and Y to pressure; label 1 is critical failure"
                  className="w-full bg-slate-950 text-xs text-slate-200 px-3 py-2 rounded-lg border border-slate-800 focus:outline-none focus:border-amber-500"
                />
              </div>

              <button
                onClick={handleAiSmartIngest}
                disabled={isAiLoading || !inputText.trim()}
                className="w-full py-2.5 px-4 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50 shadow-md"
              >
                {isAiLoading ? (
                  <>
                    <RotateCcw className="w-4 h-4 animate-spin" />
                    <span>Gemini AI is parsing and synthesizing 2D dots...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Extract & Format Data with Gemini 3.8 Flash</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* TAB 4: Classic Benchmark Datasets */}
          {activeTab === 'benchmarks' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {BENCHMARK_DATASETS.map((b) => (
                <div
                  key={b.id}
                  onClick={() => loadBenchmark(b)}
                  className="p-3.5 bg-slate-950/60 hover:bg-slate-800/60 border border-slate-800 hover:border-cyan-500/50 rounded-xl cursor-pointer transition-all space-y-1 group"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white text-xs group-hover:text-cyan-300 transition-colors">
                      {b.name}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                      {b.category}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2">{b.description}</p>
                </div>
              ))}
            </div>
          )}

          {/* Error Message banner */}
          {errorMsg && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl flex items-center gap-2 text-xs text-rose-300">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Parsed Results & Column Mapping */}
          {parsedData && (
            <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div>
                  <h4 className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span>{parsedData.name}</span>
                    <span className="text-xs font-normal text-slate-400">
                      ({parsedData.dots.length} data dots extracted)
                    </span>
                  </h4>
                  <p className="text-xs text-slate-400 mt-0.5">{parsedData.description}</p>
                </div>
              </div>

              {/* Column Mapping Controls */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1 font-medium">Feature X₁ (Horizontal):</label>
                  <select
                    value={selectedF1}
                    onChange={(e) =>
                      handleColumnMappingChange(e.target.value, selectedF2, selectedLabel, usePCA)
                    }
                    className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500"
                  >
                    {parsedData.numericColumns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1 font-medium">Feature X₂ (Vertical):</label>
                  <select
                    value={selectedF2}
                    onChange={(e) =>
                      handleColumnMappingChange(selectedF1, e.target.value, selectedLabel, usePCA)
                    }
                    className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500"
                  >
                    {parsedData.numericColumns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1 font-medium">Class / Target Column:</label>
                  <select
                    value={selectedLabel}
                    onChange={(e) =>
                      handleColumnMappingChange(selectedF1, selectedF2, e.target.value, usePCA)
                    }
                    className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="">Auto / Unlabelled</option>
                    {parsedData.columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* High-Dimensional PCA projection toggle */}
              {parsedData.numericColumns.length > 2 && (
                <div className="flex items-center justify-between p-2.5 bg-slate-900/60 border border-slate-800 rounded-lg text-xs">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    <div>
                      <span className="font-semibold text-white">
                        Multi-Feature PCA Projection ({parsedData.numericColumns.length} dimensions)
                      </span>
                      <p className="text-[11px] text-slate-400">
                        Automatically project all numerical columns down to 2 principal components
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      handleColumnMappingChange(selectedF1, selectedF2, selectedLabel, !usePCA)
                    }
                    className={`px-3 py-1 rounded-md font-semibold text-xs transition-colors ${
                      usePCA
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {usePCA ? 'PCA Active' : 'Enable 2D PCA'}
                  </button>
                </div>
              )}

              {/* Live Mini Preview & Table */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                {/* Visual Dot Scatter preview */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center">
                  <div className="w-full flex items-center justify-between text-[11px] text-slate-400 mb-2">
                    <span className="font-semibold text-slate-300">2D Normalized Dot Layout:</span>
                    <span className="font-mono tabular-nums">{parsedData.dots.length} dots</span>
                  </div>
                  <canvas
                    ref={previewCanvasRef}
                    width={280}
                    height={160}
                    className="w-full h-36 bg-slate-950 rounded-lg border border-slate-800 block"
                  />
                  <div className="w-full flex items-center justify-between text-[10px] text-slate-500 mt-1 font-mono">
                    <span>X: {selectedF1}</span>
                    <span>Y: {selectedF2}</span>
                  </div>
                </div>

                {/* Data Rows Preview */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                  <div className="text-[11px] text-slate-400 font-semibold mb-1">
                    First 4 Extracted Records:
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left font-mono text-[10px] text-slate-300">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-500">
                          <th className="py-1 px-1.5">#</th>
                          <th className="py-1 px-1.5">{selectedF1} (Norm)</th>
                          <th className="py-1 px-1.5">{selectedF2} (Norm)</th>
                          <th className="py-1 px-1.5">Class</th>
                        </tr>
                      </thead>
                      <tbody>
                        {parsedData.dots.slice(0, 4).map((d, i) => (
                          <tr key={i} className="border-b border-slate-800/40">
                            <td className="py-1 px-1.5 text-slate-500">{i + 1}</td>
                            <td className="py-1 px-1.5 tabular-nums text-cyan-300">{d.x.toFixed(3)}</td>
                            <td className="py-1 px-1.5 tabular-nums text-cyan-300">{d.y.toFixed(3)}</td>
                            <td className="py-1 px-1.5">
                              <span
                                className={`px-1 py-0.5 rounded ${
                                  d.label === 1
                                    ? 'bg-amber-500/20 text-amber-300'
                                    : 'bg-cyan-500/20 text-cyan-300'
                                }`}
                              >
                                Class {d.label}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between">
                    <span>Class 0: {parsedData.dots.filter((d) => d.label === 0).length}</span>
                    <span>Class 1: {parsedData.dots.filter((d) => d.label === 1).length}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between p-4 border-t border-slate-800 bg-slate-950/60">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            Cancel
          </button>

          <button
            onClick={handleApply}
            disabled={!parsedData || parsedData.dots.length === 0}
            className="flex items-center gap-2 px-5 py-2.5 bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-bold text-xs rounded-xl transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span>Apply to All ML Algorithms</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
