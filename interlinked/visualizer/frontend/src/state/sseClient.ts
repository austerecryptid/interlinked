// ── SSE Client — connects to backend event stream ───────────────
// Receives full snapshots on connect, then deltas for subsequent updates.

import type { SSEMessage, GraphSnapshot, GraphDelta } from '../types';

export type SSEHandler = {
  onSnapshot: (snapshot: GraphSnapshot) => void;
  onDelta: (delta: GraphDelta) => void;
  onStatus: (status: string) => void;
};

export function connectSSE(handlers: SSEHandler): () => void {
  const evtSource = new EventSource('/api/events');

  evtSource.onmessage = (event) => {
    try {
      const msg: SSEMessage = JSON.parse(event.data);
      if (msg.type === 'snapshot') {
        handlers.onSnapshot(msg.data as GraphSnapshot);
        const snap = msg.data as GraphSnapshot;
        handlers.onStatus(
          `SYNCED // ${snap.nodes?.length || 0} NODES // ${snap.edges?.length || 0} EDGES`
        );
      } else if (msg.type === 'delta') {
        handlers.onDelta(msg.data as GraphDelta);
        handlers.onStatus('DELTA APPLIED');
      }
    } catch {
      // ignore keepalive / malformed
    }
  };

  evtSource.onopen = () => handlers.onStatus('UPLINK ESTABLISHED');
  evtSource.onerror = () => handlers.onStatus('UPLINK RECONNECTING...');

  return () => evtSource.close();
}
