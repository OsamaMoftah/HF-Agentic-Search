import React from 'react';
import { useGame } from '../GameProvider.jsx';

const LANES = [
  {
    key: 'best',
    title: 'Best fits',
    description: 'Ready for a first experiment',
    filter: (dataset) => dataset.status === 'recommended' && !dataset.badges?.includes('hidden_gem'),
  },
  {
    key: 'hidden',
    title: 'Hidden gems',
    description: 'Lower adoption, strong evidence fit',
    filter: (dataset) => dataset.badges?.includes('hidden_gem'),
  },
  {
    key: 'review',
    title: 'Needs review',
    description: 'Useful, but one or more checks need attention',
    filter: (dataset) => dataset.status === 'conditional' && !dataset.badges?.includes('hidden_gem'),
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

export default function CandidateBoard() {
  const { state, selected, selectDataset } = useGame();
  const datasets = state.datasets || [];
  const visibleDatasets = datasets.filter((dataset) => dataset.status !== 'rejected');
  const rejectedCount = datasets.length - visibleDatasets.length;
  const counts = LANES.map((lane) => ({
    ...lane,
    count: visibleDatasets.filter(lane.filter).length,
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
          <div className={`decision-stat stat-${lane.key}`} key={lane.key}>
            <span>{lane.title}</span>
            <strong>{lane.count}</strong>
          </div>
        ))}
        <div className="decision-stat stat-filtered">
          <span>Filtered out</span>
          <strong>{rejectedCount}</strong>
        </div>
        <p>The board keeps attention on viable leads. Rejected datasets still appear in the ranked list when their evidence is useful.</p>
      </div>

      <div className="candidate-lanes">
        {counts.map((lane) => {
          const laneDatasets = visibleDatasets.filter(lane.filter);
          return (
            <section className={`candidate-lane lane-${lane.key}`} key={lane.key}>
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
                  {dataset.badges?.length ? (
                    <div className="candidate-badges">
                      {dataset.badges.map((badge) => (
                        <span key={badge}>{badge.replaceAll('_', ' ')}</span>
                      ))}
                    </div>
                  ) : null}
                  <p>{mainGap(dataset)}</p>
                  <small>{compactEvidence(dataset)}</small>
                  {dataset.sample_test_summary ? <small>{dataset.sample_test_summary}</small> : null}
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
                <div className="candidate-empty">No viable candidates in this tier yet.</div>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
