import React, { useState } from 'react';
import { useGame } from '../GameProvider.jsx';

const EXAMPLES = [
  'English customer-support intent data with labels for a compact classifier',
  'Arabic legal documents for abstractive summarization with a permissive license',
  'Question-answer pairs about climate science for retrieval evaluation',
];

export default function SearchPanel({ onSubmitted }) {
  const { search, thinking } = useGame();
  const [text, setText] = useState('');

  const submit = (event) => {
    event?.preventDefault();
    if (!text.trim() || thinking) return;
    search(text);
    onSubmitted?.();
  };

  return (
    <section className="search-section">
      <span className="section-index">01 / PROJECT BRIEF</span>
      <h2>Find data that actually fits the work.</h2>
      <p className="lede">
        Describe the project. The agent plans searches, inspects schemas and samples,
        checks constraints, and explains what it can verify.
      </p>
      <form onSubmit={submit}>
        <label className="field-label" htmlFor="project-brief">What are you building?</label>
        <textarea
          id="project-brief"
          rows={6}
          maxLength={2000}
          placeholder="Include task, language, modality, required fields, license, and intended use."
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) submit(event);
          }}
        />
        <div className="form-meta">
          <span>{text.length}/2000</span>
          <span>⌘ Enter to run</span>
        </div>
        <button className="primary-action" type="submit" disabled={thinking || !text.trim()}>
          <span>{thinking ? 'Researching datasets' : 'Start agentic search'}</span>
          <span aria-hidden="true">{thinking ? '•••' : '→'}</span>
        </button>
      </form>

      <div className="examples">
        <span className="field-label">Try a project</span>
        {EXAMPLES.map((example, index) => (
          <button key={example} type="button" onClick={() => setText(example)}>
            <span>0{index + 1}</span>
            {example}
          </button>
        ))}
      </div>
    </section>
  );
}
