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
          <div className="empty-weave" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </div>
          <strong>One brief. Multiple searches. Visible evidence.</strong>
          <p>The map will connect candidates whose coverage can complement each other.</p>
        </div>
      ) : null}
    </div>
  );
}
