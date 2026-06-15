import React, { useEffect, useRef } from 'react';
import { useGame } from '../GameProvider.jsx';

const LABELS = {
  started: 'Start',
  plan: 'Plan',
  search: 'Search',
  inspect: 'Inspect',
  reflect: 'Reflect',
  ranking: 'Rank',
  complete: 'Done',
  error: 'Error',
};

function traceStats(events, datasets) {
  const searches = events.filter((event) => event.type === 'search');
  const inspections = events.filter((event) => event.type === 'inspect');
  const viable = datasets.filter((dataset) => dataset.status !== 'rejected');
  const lastSearch = searches[searches.length - 1];
  const lastInspect = inspections[inspections.length - 1];
  return {
    searches: searches.length,
    inspected: inspections.length,
    viable: viable.length,
    unique: lastSearch?.unique || 0,
    current: lastInspect?.dataset_id || lastSearch?.query || '',
  };
}

function EventMeta({ event }) {
  if (event.type === 'search') {
    return (
      <div className="event-meta">
        <span>{event.found || 0} found</span>
        <span>{event.unique || 0} unique</span>
      </div>
    );
  }
  if (event.type === 'inspect') {
    return (
      <div className="event-meta">
        <span>{event.status}</span>
        <span>{event.score}/100</span>
      </div>
    );
  }
  if (event.type === 'plan' && event.queries?.length) {
    return (
      <div className="event-meta">
        <span>{event.queries.length} search angles</span>
      </div>
    );
  }
  if (event.type === 'reflect' && event.next_queries?.length) {
    return (
      <div className="event-meta">
        <span>{event.strategy}</span>
        <span>{event.next_queries.length} next angles</span>
      </div>
    );
  }
  return null;
}

export default function AgentLog() {
  const { state, events, thinking } = useGame();
  const feedRef = useRef(null);
  const stats = traceStats(events, state.datasets || []);

  useEffect(() => {
    const feed = feedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [events.length]);

  if (!events.length && !thinking) {
    return (
      <section className="process-section">
        <span className="section-index">AGENT METHOD</span>
        <p className="method-note">The trace records every search angle, inspected dataset and final ranking decision.</p>
        <ol className="method-list">
          <li><span>1</span> Parse constraints</li>
          <li><span>2</span> Search broad and niche Hub angles</li>
          <li><span>3</span> Inspect cards, schemas and samples</li>
          <li><span>4</span> Keep gems, filter weak fits</li>
        </ol>
      </section>
    );
  }

  return (
    <section className="process-section">
      <div className="process-title">
        <span className="section-index">LIVE AGENT TRACE</span>
        <span>{events.length} events</span>
      </div>
      <div className="trace-summary" aria-label="Trace summary">
        <div><span>Angles</span><strong>{stats.searches}</strong></div>
        <div><span>Unique</span><strong>{stats.unique}</strong></div>
        <div><span>Inspected</span><strong>{stats.inspected}</strong></div>
        <div><span>Viable</span><strong>{stats.viable}</strong></div>
      </div>
      {stats.current ? (
        <div className="trace-focus">
          <span>{thinking ? 'Now checking' : 'Last checked'}</span>
          <strong>{stats.current}</strong>
        </div>
      ) : null}
      <div className="event-feed" ref={feedRef}>
        {events.map((event, index) => (
          <div className={`event-row event-${event.type}`} key={`${event.type}-${index}`}>
            <span className="event-number">{String(index + 1).padStart(2, '0')}</span>
            <div>
              <strong>{LABELS[event.type] || event.type}</strong>
              <p>{event.message}</p>
              <EventMeta event={event} />
            </div>
          </div>
        ))}
        {thinking ? <div className="trace-cursor">Following the next evidence trail…</div> : null}
      </div>
    </section>
  );
}
