import React from 'react';
import { useGame } from '../GameProvider.jsx';

const STATUS_LABELS = {
  recommended: 'Recommended',
  conditional: 'Review gaps',
  rejected: 'Rejected',
};

function DatasetOrbit() {
  return (
    <div className="dataset-orbit" aria-label="Agent searches, inspects and ranks Hugging Face datasets">
      <div className="orbit-copy">
        <strong>Agent loop</strong>
        <span>Search the Hub, inspect dataset evidence, then rank what actually fits.</span>
      </div>
      <div className="orbit-stage" aria-hidden="true">
        <span className="orbit-ring ring-one" />
        <span className="orbit-ring ring-two" />
        <span className="hf-face">🤗</span>
        <span className="data-card data-card-a">
          <i />
          <b />
          <b />
        </span>
        <span className="data-card data-card-b">
          <i />
          <b />
          <b />
        </span>
        <span className="data-card data-card-c">
          <i />
          <b />
          <b />
        </span>
        <span className="trace-dot trace-dot-a" />
        <span className="trace-dot trace-dot-b" />
      </div>
    </div>
  );
}

export default function ResultsPanel() {
  const { state, selectDataset, thinking } = useGame();
  const datasets = state.datasets || [];

  return (
    <section className="results-section">
      <div className="results-heading">
        <div>
          <span className="section-index">03 / RANKED SET</span>
          <h2>Inspected datasets</h2>
        </div>
        <span>{datasets.length}</span>
      </div>

      <DatasetOrbit />

      {!datasets.length ? (
        <div className="empty-results">
          <strong>{thinking ? 'Candidates will appear here as they are inspected.' : 'No research session yet.'}</strong>
          <p>Each result includes verified checks, evidence gaps and a direct Hub link.</p>
        </div>
      ) : (
        <div className="result-list">
          {datasets.map((dataset, index) => (
            <button
              type="button"
              key={dataset.id}
              className={`result-row status-${dataset.status}`}
              onClick={() => selectDataset(dataset)}
            >
              <span className="result-rank">{String(index + 1).padStart(2, '0')}</span>
              <span className="result-main">
                <strong title={dataset.id}>{dataset.id}</strong>
                <span>{STATUS_LABELS[dataset.status] || dataset.status}</span>
              </span>
              <span className="result-score">{dataset.score}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
