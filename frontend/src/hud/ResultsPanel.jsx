import React from 'react';
import { useGame } from '../GameProvider.jsx';

const STATUS_LABELS = {
  recommended: 'Recommended',
  conditional: 'Review gaps',
  rejected: 'Rejected',
};

export default function ResultsPanel() {
  const { state, selected, selectDataset, thinking } = useGame();
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
              className={`result-row status-${dataset.status} ${selected?.id === dataset.id ? 'selected' : ''}`}
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
