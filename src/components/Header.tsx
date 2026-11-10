import React from 'react';
import { Orbit, Play, Sparkles, Layers, BookOpen, Sun, Moon, Bot, Database, Box, User as UserIcon } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface HeaderProps {
  activeTab: 'cosmos' | 'three' | 'lab' | 'journey' | 'arena';
  setActiveTab: (tab: 'cosmos' | 'three' | 'lab' | 'journey' | 'arena') => void;
  onOpenGuide: () => void;
  onOpenChat: () => void;
  onOpenIngest: () => void;
  onOpenAccount: () => void;
  isDark: boolean;
  setIsDark: (dark: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onOpenGuide,
  onOpenChat,
  onOpenIngest,
  onOpenAccount,
  isDark,
  setIsDark,
}) => {
  const { user } = useAuth();

  return (
    <header className="sticky top-0 z-40 flex items-center justify-between gap-6 px-6 py-3 border-b border-slate-800 bg-slate-950/90 backdrop-blur-md">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={() => setActiveTab('cosmos')}
          className="flex items-center gap-2.5 text-left group transition-opacity hover:opacity-90"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-indigo-500 flex items-center justify-center shadow-sm shadow-cyan-500/20">
            <Orbit className="w-4 h-4 text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight text-white whitespace-nowrap">
            DotML
          </span>
        </button>
      </div>

      {/* Zone 2: Navigation Links */}
      <nav className="hidden lg:flex items-center gap-5 text-sm font-medium text-slate-300">
        <button
          onClick={() => setActiveTab('cosmos')}
          className={`flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
            activeTab === 'cosmos'
              ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
              : 'text-slate-400 hover:text-slate-100'
          }`}
        >
          <Orbit className="w-4 h-4" />
          <span>Cosmos World</span>
        </button>

        <button
          onClick={() => setActiveTab('three')}
          className={`flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
            activeTab === 'three'
              ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
              : 'text-slate-400 hover:text-slate-100'
          }`}
        >
          <Box className="w-4 h-4 text-cyan-400" />
          <span>3D Cosmos</span>
        </button>

        <button
          onClick={() => setActiveTab('lab')}
          className={`flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
            activeTab === 'lab'
              ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
              : 'text-slate-400 hover:text-slate-100'
          }`}
        >
          <Play className="w-4 h-4" />
          <span>Algorithm Lab</span>
        </button>

        <button
          onClick={() => setActiveTab('journey')}
          className={`flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
            activeTab === 'journey'
              ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
              : 'text-slate-400 hover:text-slate-100'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>Dot Journey</span>
        </button>

        <button
          onClick={() => setActiveTab('arena')}
          className={`flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0 pb-0.5 ${
            activeTab === 'arena'
              ? 'text-cyan-400 border-b-2 border-cyan-400 font-semibold'
              : 'text-slate-400 hover:text-slate-100'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Dual Arena</span>
        </button>

        <button
          onClick={onOpenGuide}
          className="flex items-center gap-1.5 text-slate-400 hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
        >
          <BookOpen className="w-4 h-4" />
          <span>Guide</span>
        </button>
      </nav>

      {/* Zone 3: Actions & Account */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onOpenIngest}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold text-emerald-300 bg-emerald-950/50 hover:bg-emerald-900/60 border border-emerald-800/60 rounded-lg transition-colors whitespace-nowrap shrink-0 shadow-sm"
          title="Import any dataset format"
        >
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span>Import Data</span>
        </button>

        <button
          onClick={onOpenChat}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold text-cyan-300 bg-cyan-950/50 hover:bg-cyan-900/60 border border-cyan-800/60 rounded-lg transition-colors whitespace-nowrap shrink-0"
          title="Open DotML AI Mentor Chat"
        >
          <Bot className="w-3.5 h-3.5 text-cyan-400" />
          <span>AI Mentor</span>
        </button>

        {/* Firebase Cloud User Account */}
        <button
          onClick={onOpenAccount}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg transition-colors whitespace-nowrap shrink-0"
          title="Firebase Account & Cloud Datasets"
        >
          {user?.photoURL ? (
            <img src={user.photoURL} alt="User" className="w-4 h-4 rounded-full" />
          ) : (
            <UserIcon className="w-3.5 h-3.5 text-indigo-400" />
          )}
          <span>{user ? user.displayName?.split(' ')[0] || 'Account' : 'Sign In'}</span>
        </button>

        <button
          onClick={() => setIsDark(!isDark)}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-900 rounded-lg transition-colors"
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <button
          onClick={() => setActiveTab('lab')}
          className="hidden sm:inline-flex px-3.5 py-1.5 text-xs font-semibold text-white bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 rounded-lg shadow-sm shadow-cyan-500/20 transition-all whitespace-nowrap shrink-0 active:scale-95"
        >
          Launch Lab
        </button>
      </div>
    </header>
  );
};
