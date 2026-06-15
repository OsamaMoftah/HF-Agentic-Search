import React, { useState } from 'react';
import { GameProvider, useGame } from './GameProvider.jsx';
import ResearchWorkspace from './components/ResearchWorkspace.jsx';
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
          <span>HF Agentic Search</span>
        </a>
        <div className="topbar-meta">
          <span>Evidence-led dataset discovery</span>
          <span className={`run-state ${thinking ? 'active' : ''}`}>
            {thinking ? 'Agent running' : hasResults ? `${state.datasets.length} inspected` : 'Ready'}
          </span>
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="section-index">Agentic dataset research</span>
          <h1>A new way to search for datasets.</h1>
          <p>
            Describe the model or evaluation you want to build. The agent turns that brief
            into testable data requirements, searches the Hub from several angles, inspects
            schemas and samples, then shows ML engineers exactly why each dataset is useful,
            risky, or out of scope.
          </p>
        </div>
        <div className="hero-principles" aria-label="Product principles">
          <div><strong>Search wider</strong><span>Several focused queries, not one keyword lookup.</span></div>
          <div><strong>Test the fit</strong><span>Schemas, samples, language, license and access.</span></div>
          <div><strong>Show the work</strong><span>Visible evidence for recommendations and rejections.</span></div>
        </div>
      </section>

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
          <div className="workspace-heading">
            <span className="section-index">02 / AGENT WORKSPACE</span>
            {state.fallback_used ? (
              <span className="mode-note">Deterministic fallback</span>
            ) : state.model_used ? (
              <span className="mode-note">Planned with {state.model_used.split('/').pop()}</span>
            ) : null}
          </div>
          <ResearchWorkspace />
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
