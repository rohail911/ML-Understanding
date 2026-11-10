import React, { useState } from 'react';
import {
  X,
  User,
  LogOut,
  Database,
  CloudUpload,
  Trash2,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Calendar,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { DataDot } from '../types/ml';

interface UserAccountModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentDots?: DataDot[];
  currentDatasetName?: string;
  onLoadDataset: (dots: DataDot[], metadata: { name: string; description: string }) => void;
}

export const UserAccountModal: React.FC<UserAccountModalProps> = ({
  isOpen,
  onClose,
  currentDots,
  currentDatasetName,
  onLoadDataset,
}) => {
  const { user, loginWithGoogle, loginAnonymously, logout, savedDatasets, saveDataset, deleteDataset } = useAuth();
  const [saveName, setSaveName] = useState<string>('');
  const [saveDesc, setSaveDesc] = useState<string>('');
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSaveCurrent = async () => {
    if (!currentDots || currentDots.length === 0) return;
    setIsSaving(true);
    setSuccessMsg(null);
    try {
      await saveDataset({
        name: saveName.trim() || currentDatasetName || 'My ML Experiment Dataset',
        description: saveDesc.trim() || `Saved on ${new Date().toLocaleDateString()}`,
        feature1: 'Feature X',
        feature2: 'Feature Y',
        dots: currentDots,
      });
      setSuccessMsg('Dataset successfully saved to Cloud Firestore!');
      setSaveName('');
      setSaveDesc('');
    } catch (err) {
      console.error('Save failed:', err);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-800 bg-slate-950/40">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              <User className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">
                Firebase Account & Cloud Datasets
              </h3>
              <p className="text-xs text-slate-400">
                Google Authentication & Firestore persistent data storage
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

        {/* Content Body */}
        <div className="p-5 space-y-5 overflow-y-auto">
          {/* User Status Card */}
          {user ? (
            <div className="p-4 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                {user.photoURL ? (
                  <img
                    src={user.photoURL}
                    alt={user.displayName || 'User'}
                    className="w-10 h-10 rounded-full border border-cyan-500/40"
                  />
                ) : (
                  <div className="w-10 h-10 rounded-full bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold">
                    {user.displayName ? user.displayName[0].toUpperCase() : 'U'}
                  </div>
                )}
                <div>
                  <div className="font-semibold text-white text-sm flex items-center gap-1.5">
                    <span>{user.displayName || (user.isAnonymous ? 'Guest ML Explorer' : 'Signed In User')}</span>
                    <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-1.5 py-0.2 rounded border border-emerald-800/60">
                      Verified
                    </span>
                  </div>
                  <div className="text-xs text-slate-400">{user.email || 'Anonymous Guest Session'}</div>
                </div>
              </div>
              <button
                onClick={logout}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-rose-300 hover:bg-rose-950/30 border border-slate-800 hover:border-rose-900/40 rounded-lg transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <div className="p-5 bg-gradient-to-br from-cyan-950/30 to-indigo-950/30 border border-cyan-800/40 rounded-xl text-center space-y-3">
              <ShieldCheck className="w-8 h-8 text-cyan-400 mx-auto" />
              <div>
                <h4 className="text-sm font-bold text-white">Sign In with Google</h4>
                <p className="text-xs text-slate-300 max-w-sm mx-auto mt-0.5">
                  Sign in to synchronize your custom datasets and model parameters across devices in Cloud Firestore.
                </p>
              </div>
              <div className="flex justify-center gap-3 pt-1">
                <button
                  onClick={loginWithGoogle}
                  className="px-4 py-2 bg-white hover:bg-slate-100 text-slate-900 font-bold text-xs rounded-lg transition-colors shadow-md flex items-center gap-2"
                >
                  <img
                    src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
                    alt="Google"
                    className="w-4 h-4"
                  />
                  <span>Sign in with Google</span>
                </button>
                <button
                  onClick={loginAnonymously}
                  className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold rounded-lg transition-colors"
                >
                  Continue as Guest
                </button>
              </div>
            </div>
          )}

          {/* Save Active Simulation Dataset to Firestore */}
          {currentDots && currentDots.length > 0 && (
            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-semibold text-white text-xs">
                  <CloudUpload className="w-4 h-4 text-cyan-400" />
                  <span>Save Current Active Data ({currentDots.length} dots) to Firestore</span>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <input
                  type="text"
                  placeholder="Dataset Title (e.g. Spiral Epoch 4)"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-500"
                />
                <input
                  type="text"
                  placeholder="Description or notes"
                  value={saveDesc}
                  onChange={(e) => setSaveDesc(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="flex items-center justify-between">
                {successMsg ? (
                  <span className="text-xs text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>{successMsg}</span>
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-500">Stored in your personal Firestore collection</span>
                )}

                <button
                  onClick={handleSaveCurrent}
                  disabled={isSaving}
                  className="px-3 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 rounded-lg text-xs font-semibold transition-colors disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Save to Cloud'}
                </button>
              </div>
            </div>
          )}

          {/* List of User's Saved Datasets in Firestore */}
          <div className="space-y-2.5">
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center justify-between">
              <span>Your Cloud Datasets</span>
              <span className="text-slate-500 font-mono text-[10px]">{savedDatasets.length} saved</span>
            </h4>

            {savedDatasets.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
                No datasets saved in Firestore yet. Import custom data or click "Save Current Active Data" above to store your first cloud dataset!
              </div>
            ) : (
              <div className="space-y-2 max-h-52 overflow-y-auto">
                {savedDatasets.map((ds) => (
                  <div
                    key={ds.id}
                    className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center justify-between gap-3 hover:border-slate-700 transition-colors"
                  >
                    <div className="space-y-0.5">
                      <div className="font-semibold text-white text-xs flex items-center gap-2">
                        <span>{ds.name}</span>
                        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-1.5 py-0.2 rounded border border-cyan-800/60">
                          {ds.dotCount} dots
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-1">{ds.description}</p>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <button
                        onClick={() => {
                          onLoadDataset(ds.dots, { name: ds.name, description: ds.description });
                          onClose();
                        }}
                        className="px-2.5 py-1 text-xs font-semibold bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 rounded-lg transition-colors flex items-center gap-1"
                      >
                        <span>Load</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>

                      <button
                        onClick={() => deleteDataset(ds.id)}
                        className="p-1.5 text-slate-500 hover:text-rose-400 rounded-lg transition-colors"
                        title="Delete from cloud"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
