import React from 'react';
import { useGame } from '../GameProvider.jsx';

function Check({ label, value }) {
  return (
    <div className={`check-row check-${value}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SampleTest({ test }) {
  return (
    <div className={`sample-test sample-${test.status}`}>
      <strong>{test.name}</strong>
      <span>{test.status}</span>
      <p>{test.detail}</p>
    </div>
  );
}

export default function DetailCard() {
  const { selected } = useGame();
  if (!selected) return null;

  return (
    <article className={`detail-card status-${selected.status}`}>
      <div className="detail-kicker">
        <span>{selected.status}</span>
        <strong>{selected.score}/100</strong>
      </div>
      <h2>{selected.id.split('/').pop()}</h2>
      <a className="dataset-id" href={selected.hub_url} target="_blank" rel="noreferrer">
        {selected.id} ↗
      </a>
      <p className="detail-description">
        {selected.description || 'No dataset-card description was available.'}
      </p>
      {selected.discovery_note ? <p className="discovery-note">{selected.discovery_note}</p> : null}

      <div className="score-grid">
        {Object.entries(selected.score_breakdown || {}).map(([key, value]) => (
          <div key={key}>
            <span>{key.replaceAll('_', ' ')}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <div className="detail-block">
        <span className="field-label">Constraint checks</span>
        <div className="check-grid">
          {Object.entries(selected.checks || {}).map(([key, value]) => (
            <Check key={key} label={key.replaceAll('_', ' ')} value={value} />
          ))}
        </div>
      </div>

      <div className="detail-block">
        <span className="field-label">Inspected evidence</span>
        <ul className="evidence-list">
          {(selected.evidence || []).map((item) => <li key={item}>{item}</li>)}
        </ul>
      </div>

      {selected.features?.length ? (
        <div className="detail-block">
          <span className="field-label">Schema fields</span>
          <div className="tag-list">
            {selected.features.slice(0, 14).map((feature) => <span key={feature}>{feature}</span>)}
          </div>
        </div>
      ) : null}

      {selected.sample_tests?.length ? (
        <div className="detail-block">
          <span className="field-label">Sample tests</span>
          <div className="sample-test-list">
            {selected.sample_tests.map((test) => <SampleTest key={test.name} test={test} />)}
          </div>
        </div>
      ) : null}

      {selected.loader_snippet ? (
        <div className="detail-block">
          <span className="field-label">Starter loader</span>
          <pre className="loader-snippet">{selected.loader_snippet}</pre>
        </div>
      ) : null}

      <div className="verdict">
        <strong>{selected.recommendation}</strong>
        <p>{selected.weakness}</p>
      </div>
    </article>
  );
}
