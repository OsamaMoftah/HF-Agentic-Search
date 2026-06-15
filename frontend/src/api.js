const SESSION_KEY = 'hf_agentic_search_session';

export function sessionId() {
  let sid = sessionStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = `dw-${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`;
    sessionStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.error || `Request failed with HTTP ${response.status}`;
  } catch {
    return `Request failed with HTTP ${response.status}`;
  }
}

export async function weaveQueryStream(task, onEvent, signal) {
  const response = await fetch('/weave/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId() },
    body: JSON.stringify({ task }),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  if (!response.body) throw new Error('Streaming is not supported by this browser.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

export async function getState() {
  const response = await fetch('/state', { headers: { 'X-Session-Id': sessionId() } });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}
