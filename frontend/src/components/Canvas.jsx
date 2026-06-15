import React, { useEffect, useRef } from 'react';
import { useGame } from '../GameProvider.jsx';
import { WeaverScene } from '../pixi/WeaverScene.js';

export default function Canvas() {
  const { state, selected, selectDataset } = useGame();
  const hostRef = useRef(null);
  const sceneRef = useRef(null);
  const datasetsRef = useRef(state.datasets);
  const selectRef = useRef(selectDataset);

  datasetsRef.current = state.datasets;
  selectRef.current = selectDataset;

  useEffect(() => {
    const scene = new WeaverScene();
    sceneRef.current = scene;
    scene.init(hostRef.current, {
      onNodeTap: (id) => {
        const dataset = datasetsRef.current.find((item) => item.id === id);
        if (dataset) selectRef.current(dataset);
      },
    });
    return () => scene.destroy();
  }, []);

  useEffect(() => {
    sceneRef.current?.render(state, selected?.id);
  }, [state, selected?.id]);

  return (
    <div className="canvas-host" ref={hostRef}>
      {!state.nodes?.length ? (
        <div className="canvas-empty">
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
      ) : null}
    </div>
  );
}
