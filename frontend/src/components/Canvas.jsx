import React from 'react';
import { useGame } from '../GameProvider.jsx';

const LANES = [
  {
    status: 'recommended',
    title: 'Recommended',
    description: 'Ready for a first experiment',
  },
  {
    status: 'conditional',
    title: 'Review gaps',
    description: 'Useful, but one or more checks need attention',
  },
  {
    status: 'rejected',
    title: 'Rejected',
    description: 'Does not satisfy this project brief',
  },
];

const CHECKS = [
  ['domain', 'Domain'],
  ['required_fields', 'Schema'],
  ['language', 'Lang'],
  ['license', 'License'],
  ['accessible', 'Access'],
];

function shortName(id) {
  return id?.split('/').pop()?.replaceAll('_', ' ') || id;
}

function mainGap(dataset) {
  if (dataset.status === 'recommended') return 'All critical evidence checks passed.';
  if (dataset.rejection_reasons?.length) return dataset.rejection_reasons[0];
  return dataset.weakness || 'Review the remaining evidence gaps before using it.';
}

function compactEvidence(dataset) {
  const features = dataset.features?.slice(0, 3).join(', ');
  const examples = dataset.num_examples ? `${dataset.num_examples.toLocaleString()} rows` : 'size unknown';
  return features ? `${features} · ${examples}` : examples;
}

export default function Canvas() {
  const { state, selected, selectDataset } = useGame();
  const datasets = state.datasets || [];
  const counts = LANES.map((lane) => ({
    ...lane,
    count: datasets.filter((dataset) => dataset.status === lane.status).length,
  }));

  if (!datasets.length) {
    return (
      <div className="candidate-board empty">
        <div className="workflow-preview" aria-hidden="true">
          <div><span>01</span><strong>Search</strong><small>Multiple Hub queries</small></div>
          <i />
          <div><span>02</span><strong>Test</strong><small>Schema and sample checks</small></div>
          <i />
          <div><span>03</span><strong>Explain</strong><small>Documented evidence</small></div>
        </div>
        <strong>Every candidate must earn its place.</strong>
        <p>The agent searches broadly, then narrows the field by checking the actual dataset, not just its title.</p>
      </div>
    );
  }

  return (
    <div className="candidate-board">
      <div className="decision-summary">
        {counts.map((lane) => (
          <div className={`decision-stat stat-${lane.status}`} key={lane.status}>
            <span>{lane.title}</span>
            <strong>{lane.count}</strong>
          </div>
        ))}
        <p>Read left to right as a decision queue: usable now, usable with caveats, or not a fit for this brief.</p>
      </div>

      <div className="candidate-lanes">
        {counts.map((lane) => {
          const laneDatasets = datasets.filter((dataset) => dataset.status === lane.status);
          return (
            <section className={`candidate-lane lane-${lane.status}`} key={lane.status}>
              <div className="lane-heading">
                <div>
                  <strong>{lane.title}</strong>
                  <span>{lane.description}</span>
                </div>
                <b>{lane.count}</b>
              </div>

              {laneDatasets.length ? laneDatasets.map((dataset) => (
                <button
                  type="button"
                  className={`candidate-card ${selected?.id === dataset.id ? 'active' : ''}`}
                  key={dataset.id}
                  onClick={() => selectDataset(dataset)}
                >
                  <div className="candidate-card-top">
                    <strong title={dataset.id}>{shortName(dataset.id)}</strong>
                    <span>{dataset.score}</span>
                  </div>
                  <p>{mainGap(dataset)}</p>
                  <small>{compactEvidence(dataset)}</small>
                  <div className="check-dots" aria-label="Evidence checks">
                    {CHECKS.map(([key, label]) => {
                      const value = dataset.checks?.[key] || 'unknown';
                      return (
                        <span className={`check-dot check-${value}`} key={key} title={`${label}: ${value}`}>
                          {label}
                        </span>
                      );
                    })}
                  </div>
                </button>
              )) : (
                <div className="candidate-empty">No candidates in this tier yet.</div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
