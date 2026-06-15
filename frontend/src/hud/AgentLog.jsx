import React, { useEffect, useRef } from 'react';
import { useGame } from '../GameProvider.jsx';

const LABELS = {
  started: 'Start',
  plan: 'Plan',
  search: 'Search',
  inspect: 'Inspect',
  ranking: 'Rank',
  complete: 'Done',
  error: 'Error',
};

export default function AgentLog() {
  const { events, thinking } = useGame();
  const feedRef = useRef(null);

  useEffect(() => {
    const feed = feedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [events.length]);

  if (!events.length && !thinking) {
    return (
      <section className="process-section">
        <span className="section-index">AGENT METHOD</span>
        <ol className="method-list">
          <li><span>1</span> Parse constraints</li>
          <li><span>2</span> Search multiple angles</li>
          <li><span>3</span> Inspect cards, schemas and samples</li>
          <li><span>4</span> Rank with visible evidence</li>
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
      <div className="event-feed" ref={feedRef}>
        {events.map((event, index) => (
          <div className={`event-row event-${event.type}`} key={`${event.type}-${index}`}>
            <span className="event-number">{String(index + 1).padStart(2, '0')}</span>
            <div>
              <strong>{LABELS[event.type] || event.type}</strong>
              <p>{event.message}</p>
            </div>
          </div>
        ))}
        {thinking ? <div className="trace-cursor">Inspecting the next signal…</div> : null}
      </div>
    </section>
  );
}
