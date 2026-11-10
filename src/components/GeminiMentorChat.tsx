import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, X, Sparkles, AlertCircle, RefreshCw, ChevronDown } from 'lucide-react';
import { AlgorithmMetadata } from '../types/ml';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface GeminiMentorChatProps {
  isOpen: boolean;
  onClose: () => void;
  currentAlgo?: AlgorithmMetadata;
  currentMetrics?: any;
}

export const GeminiMentorChat: React.FC<GeminiMentorChatProps> = ({
  isOpen,
  onClose,
  currentAlgo,
  currentMetrics,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `Hello! I am your DotML AI Mentor. Ask me anything about how the data dots, decision boundaries, or hyperparameters behave in real time. Try adjusting parameters on the canvas and ask me why the model changes!`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<'gemini-3.8-flash' | 'gemini-3.1-flash-lite'>('gemini-3.8-flash');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || input).trim();
    if (!text || isLoading) return;

    setErrorMsg(null);
    const userMsg: ChatMessage = {
      id: `usr-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newHistory.map((m) => ({ role: m.role, content: m.content })),
          model: selectedModel,
          currentContext: {
            algorithm: currentAlgo?.name,
            category: currentAlgo?.category,
            formula: currentAlgo?.mathFormula,
            metrics: currentMetrics,
          },
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      const assistantMsg: ChatMessage = {
        id: `asst-${Date.now()}`,
        role: 'assistant',
        content: data.reply || 'No response received from the mentor.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      console.error('Chat Error:', err);
      setErrorMsg(err?.message || 'Failed to reach AI mentor.');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96 sm:w-[420px] h-[540px] bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-200">
      {/* Top Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-500 flex items-center justify-center text-white shadow-sm">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h3 className="text-sm font-bold text-white tracking-tight">DotML AI Mentor</h3>
              <span className="text-[10px] text-cyan-400 font-medium bg-cyan-950/60 border border-cyan-800/40 px-1.5 py-0.2 rounded">
                Gemini
              </span>
            </div>
            <div className="text-[10px] text-slate-400">
              Observing {currentAlgo?.name || 'Algorithm Cosmos'}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Model Selector */}
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value as any)}
            className="text-[10px] bg-slate-900 text-slate-300 border border-slate-800 rounded px-1.5 py-1 font-mono focus:outline-none"
            title="Switch Gemini Model"
          >
            <option value="gemini-3.8-flash">3.8 Flash (Deep)</option>
            <option value="gemini-3.1-flash-lite">3.1 Lite (Fast)</option>
          </select>

          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div ref={scrollRef} className="flex-1 p-4 overflow-y-auto space-y-3.5 bg-slate-950/50 text-xs">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 leading-relaxed shadow-sm ${
                msg.role === 'user'
                  ? 'bg-cyan-600 text-white rounded-br-none'
                  : 'bg-slate-800 text-slate-200 border border-slate-700/60 rounded-bl-none'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
            </div>
            <span className="text-[10px] text-slate-500 mt-1 px-1">{msg.timestamp}</span>
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 p-3 bg-slate-800/60 border border-slate-700/40 rounded-xl text-slate-300 w-fit">
            <Sparkles className="w-4 h-4 text-cyan-400 animate-spin" />
            <span className="text-xs">Analyzing dot mathematics...</span>
          </div>
        )}

        {errorMsg && (
          <div className="p-3 bg-rose-950/30 border border-rose-800/50 rounded-xl text-rose-300 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="text-xs leading-relaxed">
              <span>{errorMsg}</span>
            </div>
          </div>
        )}
      </div>

      {/* Quick Prompt Chips */}
      <div className="px-3 py-1.5 bg-slate-950 border-t border-slate-800/80 flex items-center gap-1.5 overflow-x-auto text-[11px]">
        <span className="text-slate-500 whitespace-nowrap text-[10px]">Ask:</span>
        <button
          onClick={() => handleSend(`Why does tuning this parameter change the boundary?`)}
          className="text-slate-300 hover:text-white bg-slate-900 border border-slate-800 rounded px-2 py-0.5 whitespace-nowrap hover:bg-slate-800 transition-colors"
        >
          Explain Boundary
        </button>
        <button
          onClick={() => handleSend(`What does this loss value mean?`)}
          className="text-slate-300 hover:text-white bg-slate-900 border border-slate-800 rounded px-2 py-0.5 whitespace-nowrap hover:bg-slate-800 transition-colors"
        >
          Interpret Loss
        </button>
        <button
          onClick={() => handleSend(`How do the support vectors or centroids work?`)}
          className="text-slate-300 hover:text-white bg-slate-900 border border-slate-800 rounded px-2 py-0.5 whitespace-nowrap hover:bg-slate-800 transition-colors"
        >
          Key Dots Role
        </button>
      </div>

      {/* Input Field */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="p-3 bg-slate-950 border-t border-slate-800 flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask AI Mentor about this simulation..."
          className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="p-2 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-40 text-white rounded-xl transition-all shadow-sm"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
