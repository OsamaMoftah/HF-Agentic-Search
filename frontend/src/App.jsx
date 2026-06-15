import React, { useState } from 'react';
import { GameProvider, useGame } from './GameProvider.jsx';
import Canvas from './components/Canvas.jsx';
import SearchPanel from './hud/SearchPanel.jsx';
import ResultsPanel from './hud/ResultsPanel.jsx';
import DetailCard from './hud/DetailCard.jsx';
import AgentLog from './hud/AgentLog.jsx';

function Workspace() {
  const { state, thinking } = useGame();
  const [mobileTab, setMobileTab] = useState('search');
  const hasResults = state.datasets.length > 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="HF Agentic Search home">
          <span className="brand-mark" aria-hidden="true">HFA</span>
          <span>HF Agentic Search</span>
        </a>
        <div className="topbar-meta">
          <span>Evidence-led dataset discovery</span>
          <span className={`run-state ${thinking ? 'active' : ''}`}>
            {thinking ? 'Agent running' : hasResults ? `${state.datasets.length} inspected` : 'Ready'}
          </span>
        </div>
      </header>

      <nav className="mobile-tabs" aria-label="Workspace views">
        {['search', 'map', 'results'].map((tab) => (
          <button
            key={tab}
            className={mobileTab === tab ? 'active' : ''}
            onClick={() => setMobileTab(tab)}
            type="button"
          >
            {tab}
          </button>
        ))}
      </nav>

      <div className="workspace">
        <aside className={`rail rail-left ${mobileTab === 'search' ? 'mobile-active' : ''}`}>
          <SearchPanel onSubmitted={() => setMobileTab('map')} />
          <AgentLog />
        </aside>

        <main className={`map-stage ${mobileTab === 'map' ? 'mobile-active' : ''}`}>
          <div className="map-heading">
            <div>
              <span className="section-index">02 / EVIDENCE MAP</span>
              <h2>{hasResults ? 'Candidates, connected' : 'Search the Hub with a research agent'}</h2>
            </div>
            {state.fallback_used ? (
              <span className="mode-note">Deterministic fallback</span>
            ) : state.model_used ? (
              <span className="mode-note">Planned with {state.model_used.split('/').pop()}</span>
            ) : null}
          </div>
          <Canvas />
        </main>

        <aside className={`rail rail-right ${mobileTab === 'results' ? 'mobile-active' : ''}`}>
          <DetailCard />
          <ResultsPanel />
        </aside>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <GameProvider>
      <Workspace />
    </GameProvider>
  );
}
