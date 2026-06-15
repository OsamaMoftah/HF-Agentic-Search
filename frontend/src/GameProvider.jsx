import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { getState, weaveQueryStream } from './api.js';

const GameContext = createContext(null);

const EMPTY_STATE = {
  datasets: [],
  nodes: [],
  threads: [],
  task: '',
  top_pick: null,
};

export function GameProvider({ children }) {
  const [state, setState] = useState(EMPTY_STATE);
  const [events, setEvents] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [selected, setSelected] = useState(null);
  const abortRef = useRef(null);

  const selectDataset = useCallback((dataset) => setSelected(dataset), []);

  const search = useCallback(async (task) => {
    const normalized = task.trim();
    if (!normalized) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setThinking(true);
    setSelected(null);
    setEvents([]);
    setState({ ...EMPTY_STATE, task: normalized });

    try {
      await weaveQueryStream(normalized, (event) => {
        if (event.type !== 'candidate') {
          setEvents((current) => [...current, event]);
        }
        if (event.type === 'candidate') {
          setState((current) => ({
            ...current,
            datasets: [...current.datasets, event.dataset],
          }));
        }
        if (event.type === 'complete') {
          setState(event.result);
          const top = event.result.datasets.find((dataset) => dataset.id === event.result.top_pick);
          setSelected(top || event.result.datasets[0] || null);
        }
        if (event.type === 'error') throw new Error(event.message);
      }, controller.signal);
    } catch (error) {
      if (error.name !== 'AbortError') {
        setEvents((current) => [...current, { type: 'error', message: error.message }]);
      }
    } finally {
      if (abortRef.current === controller) {
        setThinking(false);
        abortRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    let active = true;
    getState()
      .then((saved) => {
        if (!active || !saved?.datasets?.length) return;
        setState(saved);
        const top = saved.datasets.find((dataset) => dataset.id === saved.top_pick);
        setSelected(top || saved.datasets[0]);
      })
      .catch(() => {});
    return () => {
      active = false;
      abortRef.current?.abort();
    };
  }, []);

  const value = useMemo(
    () => ({ state, events, thinking, selected, selectDataset, search }),
    [state, events, thinking, selected, selectDataset, search],
  );
  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame() {
  const context = useContext(GameContext);
  if (!context) throw new Error('useGame must be used inside GameProvider');
  return context;
}
