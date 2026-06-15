import React, { useMemo } from 'react';
import { useGame } from '../GameProvider.jsx';
import CandidateBoard from './CandidateBoard.jsx';

const STEPS = [
  { type: 'plan', label: 'Understand', detail: 'Turn the brief into explicit requirements' },
  { type: 'search', label: 'Search', detail: 'Explore several angles across the Hub' },
  { type: 'inspect', label: 'Inspect', detail: 'Read cards, schemas and sample rows' },
  { type: 'reflect', label: 'Reflect', detail: 'Revise the search from evidence gaps' },
  { type: 'ranking', label: 'Rank', detail: 'Compare evidence and explain the result' },
];

const CHECKS = [
  ['domain', 'Domain', 'Does it cover the intended subject?'],
  ['required_fields', 'Schema', 'Are the required fields really present?'],
  ['language', 'Language', 'Does the content match the requested language?'],
  ['license', 'License', 'Can the project use it as intended?'],
  ['accessible', 'Access', 'Can the agent inspect and load it?'],
];

function formatValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'Not specified';
  return value || 'Not specified';
}

export default function ResearchWorkspace() {
  const { state, events, thinking } = useGame();
  const profile = state.profile || [...events].reverse().find((event) => event.type === 'plan')?.profile;
  const completedTypes = new Set(events.map((event) => event.type));
  const activeStep = thinking
    ? Math.max(0, STEPS.findIndex((step) => !completedTypes.has(step.type)))
    : state.datasets.length
      ? STEPS.length - 1
      : -1;

  const checkSummary = useMemo(() => CHECKS.map(([key, label, detail]) => {
    const values = state.datasets.map((dataset) => dataset.checks?.[key]).filter(Boolean);
    const pass = values.filter((value) => value === 'pass').length;
    const fail = values.filter((value) => value === 'fail').length;
    return { key, label, detail, pass, fail, total: values.length };
  }), [state.datasets]);

  const briefItems = [
    ['Task', profile?.task_type],
    ['Language', profile?.languages],
    ['Modality', profile?.modalities],
    ['Required schema', profile?.required_fields],
    ['License', profile?.license],
  ];

  return (
    <>
      <section className="brief-strip" aria-label="Interpreted project brief">
        <div className="brief-strip-heading">
          <span className="section-index">Interpreted project brief</span>
          <span>{profile ? 'Verified from your request' : 'Appears after you start a search'}</span>
        </div>
        <div className="brief-grid">
          {briefItems.map(([label, value]) => (
            <div className="brief-item" key={label}>
              <span>{label}</span>
              <strong>{formatValue(value)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="agent-progress" aria-label="Agent progress">
        {STEPS.map((step, index) => {
          const completed = completedTypes.has(step.type) || (!thinking && state.datasets.length && index < 3);
          const active = index === activeStep;
          return (
            <div className={`progress-step ${completed ? 'complete' : ''} ${active ? 'active' : ''}`} key={step.type}>
              <span className="progress-number">{completed ? '✓' : index + 1}</span>
              <div>
                <strong>{step.label}</strong>
                <p>{step.detail}</p>
              </div>
            </div>
          );
        })}
      </section>

      <div className="research-grid">
        <section className="evidence-panel">
          <div className="panel-heading">
            <div>
              <span className="section-index">Decision board</span>
              <h2>{state.datasets.length ? `${state.datasets.length} inspected candidates` : 'How the search becomes evidence'}</h2>
            </div>
            <span className="panel-count">{state.datasets.length}</span>
          </div>
          <CandidateBoard />
        </section>

        <aside className="check-panel">
          <div className="panel-heading">
            <div>
              <span className="section-index">Evidence coverage</span>
              <h2>What the agent checks</h2>
            </div>
          </div>
          <div className="check-summary">
            {checkSummary.map((check) => (
              <div className="check-summary-row" key={check.key}>
                <div>
                  <strong>{check.label}</strong>
                  <p>{check.detail}</p>
                </div>
                <span className={check.fail ? 'has-fail' : check.pass ? 'has-pass' : ''}>
                  {check.total ? `${check.pass}/${check.total}` : 'Pending'}
                </span>
              </div>
            ))}
          </div>
          <div className="honesty-note">
            <strong>Honest by design</strong>
            <p>Unknown metadata stays unknown. Missing evidence lowers confidence instead of becoming a guessed score.</p>
          </div>
        </aside>
      </div>
    </>
  );
}
