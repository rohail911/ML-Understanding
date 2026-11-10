import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { CosmosWorld } from './components/CosmosWorld';
import { ThreeCosmosScene } from './components/ThreeCosmosScene';
import { AlgorithmSandbox } from './components/AlgorithmSandbox';
import { DotJourney } from './components/DotJourney';
import { DualArena } from './components/DualArena';
import { GuideModal } from './components/GuideModal';
import { GeminiMentorChat } from './components/GeminiMentorChat';
import { DataIngestionModal } from './components/DataIngestionModal';
import { UserAccountModal } from './components/UserAccountModal';
import { AlgorithmId, DataDot } from './types/ml';
import { ALGORITHMS } from './utils/algorithms';

export default function App() {
  const [activeTab, setActiveTab] = useState<'cosmos' | 'three' | 'lab' | 'journey' | 'arena'>('cosmos');
  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmId>('kmeans');
  const [isGuideOpen, setIsGuideOpen] = useState<boolean>(false);
  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
  const [isIngestOpen, setIsIngestOpen] = useState<boolean>(false);
  const [isAccountOpen, setIsAccountOpen] = useState<boolean>(false);
  const [isDark, setIsDark] = useState<boolean>(true);

  // Custom User-Imported Dataset
  const [customDataset, setCustomDataset] = useState<{
    dots: DataDot[];
    name: string;
    description: string;
  } | null>(null);

  // Sync theme with HTML class
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  // When an algorithm is clicked in the Cosmos World, navigate straight to its Interactive Lab
  const handleSelectFromCosmos = (algoId: AlgorithmId) => {
    setSelectedAlgo(algoId);
    setActiveTab('lab');
  };

  // When custom data is ingested via file upload, paste, or AI Smart Ingest
  const handleApplyDataset = (dots: DataDot[], metadata: { name: string; description: string }) => {
    setCustomDataset({
      dots,
      name: metadata.name,
      description: metadata.description,
    });
    // Immediately open the interactive lab to run algorithms on the new dataset!
    setActiveTab('lab');
  };

  const activeAlgoMeta = ALGORITHMS.find((a) => a.id === selectedAlgo);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans select-none">
      {/* Top Navigation Bar adhering to Top Bar Contract */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenGuide={() => setIsGuideOpen(true)}
        onOpenChat={() => setIsChatOpen((c) => !c)}
        onOpenIngest={() => setIsIngestOpen(true)}
        onOpenAccount={() => setIsAccountOpen(true)}
        isDark={isDark}
        setIsDark={setIsDark}
      />

      {/* Main Interactive Stage */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        {activeTab === 'cosmos' && (
          <CosmosWorld onSelectAlgorithm={handleSelectFromCosmos} />
        )}

        {activeTab === 'three' && (
          <ThreeCosmosScene
            dots={customDataset?.dots}
            customDatasetName={customDataset?.name}
            onOpenIngestModal={() => setIsIngestOpen(true)}
          />
        )}

        {activeTab === 'lab' && (
          <AlgorithmSandbox
            algorithmId={selectedAlgo}
            onSelectAlgorithm={(id) => setSelectedAlgo(id)}
            onOpenChat={() => setIsChatOpen(true)}
            customDots={customDataset?.dots}
            customDatasetName={customDataset?.name}
            onOpenIngestModal={() => setIsIngestOpen(true)}
            onClearCustomDataset={() => setCustomDataset(null)}
          />
        )}

        {activeTab === 'journey' && <DotJourney />}

        {activeTab === 'arena' && (
          <DualArena
            customDots={customDataset?.dots}
            customDatasetName={customDataset?.name}
            onOpenIngestModal={() => setIsIngestOpen(true)}
          />
        )}
      </main>

      {/* Exploration & Architecture Guide Modal */}
      <GuideModal isOpen={isGuideOpen} onClose={() => setIsGuideOpen(false)} />

      {/* Universal Data Ingestion Modal (Any Format -> Any ML Algorithm) */}
      <DataIngestionModal
        isOpen={isIngestOpen}
        onClose={() => setIsIngestOpen(false)}
        onApplyDataset={handleApplyDataset}
      />

      {/* Firebase Cloud User Account & Datasets Modal */}
      <UserAccountModal
        isOpen={isAccountOpen}
        onClose={() => setIsAccountOpen(false)}
        currentDots={customDataset?.dots}
        currentDatasetName={customDataset?.name}
        onLoadDataset={(dots, meta) => {
          setCustomDataset({ dots, name: meta.name, description: meta.description });
          setActiveTab('lab');
        }}
      />

      {/* Multi-Turn Gemini AI Mentor Chat Interface */}
      <GeminiMentorChat
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        currentAlgo={activeAlgoMeta}
      />
    </div>
  );
}
