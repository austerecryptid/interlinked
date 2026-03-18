// ── INTERLINKED — Main Application Shell ────────────────────────
// WebGL graph explorer with Blade Runner / Tron cyberpunk aesthetic.

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import Graph from 'graphology';
import { createGraphStore, applySnapshot, applyDelta } from './state/graphStore';
import { connectSSE } from './state/sseClient';
import GraphCanvas from './graph/GraphCanvas';
import { NODE_COLORS, NODE_SHAPES, EDGE_COLORS, EDGE_DASH, EDGE_WIDTH, TRACE_ROLE_COLORS, TRACE_EDGE_COLORS, GOSLING_QUOTES, RADIO_STATIONS } from './theme';
import type { Stats, NodeData, GraphSnapshot, GraphDelta, ViewState } from './types';

const store = createGraphStore();

export default function App() {
  const [, forceRender] = useState(0);
  const rerender = useCallback(() => forceRender((n) => n + 1), []);

  const [stats, setStats] = useState<Stats | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState('ESTABLISHING UPLINK...');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = useState(-1);
  const [rightTab, setRightTab] = useState<'inspect' | 'chat' | 'settings'>('inspect');
  const [chatMessages, setChatMessages] = useState<{ role: string; content: string; commands?: string[] }[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);
  const [llmModel, setLlmModel] = useState('claude-sonnet-4-20250514');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [nodeGlow, setNodeGlow] = useState(0.5);
  const [edgeGlow, setEdgeGlow] = useState(0.5);

  const commandRef = useRef<HTMLInputElement>(null!);
  const chatInputRef = useRef<HTMLTextAreaElement>(null!);
  const chatScrollRef = useRef<HTMLDivElement>(null!);

  // SSE connection + initial data
  useEffect(() => {
    fetch('/api/stats').then((r) => r.json()).then(setStats).catch(() => {});
    fetch('/api/settings')
      .then((r) => r.json())
      .then((data) => {
        setHasApiKey(data.has_api_key);
        setLlmModel(data.model);
      })
      .catch(() => {});

    const disconnect = connectSSE({
      onSnapshot: (snapshot: GraphSnapshot) => {
        applySnapshot(store, snapshot);
        // Sync React state if store cleared the stale selection
        setSelectedNode(store.selectedNodeId);
        rerender();
        fetch('/api/stats').then((r) => r.json()).then(setStats).catch(() => {});
      },
      onDelta: (delta: GraphDelta) => {
        applyDelta(store, delta);
        // Sync React state if store cleared the stale selection
        setSelectedNode(store.selectedNodeId);
        rerender();
      },
      onStatus: (status: string) => setStatusMsg(status),
    });

    return disconnect;
  }, [rerender]);

  // Auto-scroll chat
  useEffect(() => {
    if (chatScrollRef.current) chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
  }, [chatMessages]);

  const refreshView = useCallback(() => {
    fetch('/api/snapshot')
      .then((r) => r.json())
      .then((data: GraphSnapshot) => {
        applySnapshot(store, data);
        setSelectedNode(store.selectedNodeId);
        rerender();
      });
    fetch('/api/stats').then((r) => r.json()).then(setStats).catch(() => {});
  }, [rerender]);

  // Command bar
  const sendCommand = useCallback(
    (cmd: string) => {
      if (!cmd.trim()) return;
      setCommandHistory((h) => [cmd, ...h].slice(0, 50));
      setHistoryIdx(-1);
      const isPython = cmd.includes('view.') || cmd.includes('graph.') || cmd.includes('=') || cmd.includes('(');
      const endpoint = isPython ? '/api/command' : '/api/nl';
      const body = isPython ? { command: cmd } : { text: cmd };

      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.error) {
            setStatusMsg(`ERR: ${data.error}`.toUpperCase());
          } else {
            const msg = data.result || JSON.stringify(data).slice(0, 200);
            setStatusMsg(typeof msg === 'string' ? msg.toUpperCase() : String(msg).toUpperCase());
            refreshView();
          }
        });
    },
    [refreshView],
  );

  // Chat
  const sendChat = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      setChatMessages((msgs) => [...msgs, { role: 'user', content: text }]);
      setChatLoading(true);

      fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
        .then((r) => r.json())
        .then((data) => {
          setChatLoading(false);
          if (data.error === 'no_api_key') {
            setChatMessages((msgs) => [
              ...msgs,
              {
                role: 'system',
                content:
                  'No API key configured. Go to Settings tab to add your Anthropic API key, or use the command bar with Python commands.',
              },
            ]);
            return;
          }
          setChatMessages((msgs) => [
            ...msgs,
            { role: 'assistant', content: data.explanation || 'Done.', commands: data.commands_run },
          ]);
          refreshView();
        })
        .catch((err) => {
          setChatLoading(false);
          setChatMessages((msgs) => [...msgs, { role: 'system', content: `Error: ${err.message}` }]);
        });
    },
    [refreshView],
  );

  const handleZoom = useCallback(
    (level: string) => {
      fetch('/api/zoom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level }),
      })
        .then((r) => r.json())
        .then((data) => {
          setStatusMsg(data.result?.toUpperCase() || '');
          refreshView();
        });
    },
    [refreshView],
  );

  const handleNodeClick = useCallback((nodeId: string) => {
    const id = nodeId || null;
    store.selectedNodeId = id;  // sync write — Sigma reads this immediately
    setSelectedNode(id);
    if (nodeId) setRightTab('inspect');
  }, []);

  const handleNodeDoubleClick = useCallback(
    (nodeId: string) => {
      fetch('/api/focus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, depth: 2 }),
      })
        .then((r) => r.json())
        .then((data) => {
          setStatusMsg(data.result?.toUpperCase() || '');
          refreshView();
        });
    },
    [refreshView],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        sendCommand((e.target as HTMLInputElement).value);
        (e.target as HTMLInputElement).value = '';
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setHistoryIdx((i) => {
          const next = Math.min(i + 1, commandHistory.length - 1);
          if (commandRef.current && commandHistory[next]) commandRef.current.value = commandHistory[next];
          return next;
        });
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setHistoryIdx((i) => {
          const next = Math.max(i - 1, -1);
          if (commandRef.current) commandRef.current.value = next >= 0 ? commandHistory[next] : '';
          return next;
        });
      }
    },
    [sendCommand, commandHistory],
  );

  const handleChatKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat((e.target as HTMLTextAreaElement).value);
        (e.target as HTMLTextAreaElement).value = '';
      }
    },
    [sendChat],
  );

  const handleSaveSettings = useCallback((apiKey: string, model: string) => {
    fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey, model }),
    })
      .then((r) => r.json())
      .then((data) => {
        setHasApiKey(data.has_api_key);
        setLlmModel(data.model);
        setStatusMsg(data.has_api_key ? 'API KEY CONFIGURED' : 'API KEY CLEARED');
      });
  }, []);

  const currentZoom = store.viewState?.zoom_level || 'module';
  const vs = store.viewState;
  const hasActiveFilters = !!(
    (vs?.highlighted_node_ids?.length) ||
    (vs?.visible_node_ids?.length) ||
    (vs?.filter_expression) ||
    (vs?.trace_node_roles && Object.keys(vs.trace_node_roles).length > 0)
  );
  const activeEdgeTypes = store.viewState?.visible_edge_types || [
    'calls', 'imports', 'inherits', 'reads', 'writes', 'contains', 'returns',
  ];

  const handleEdgeToggle = useCallback(
    (edgeType: string) => {
      const current = new Set(activeEdgeTypes);
      if (current.has(edgeType)) current.delete(edgeType);
      else current.add(edgeType);
      fetch('/api/edge_types', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edge_types: [...current] }),
      }).then(() => refreshView());
    },
    [activeEdgeTypes, refreshView],
  );

  // Selected node data from graphology
  const selectedNodeData = useMemo(() => {
    if (!selectedNode || !store.graph.hasNode(selectedNode)) return null;
    return store.graph.getNodeAttributes(selectedNode) as Record<string, unknown>;
  }, [selectedNode, store.graph]);

  const gridCols = `${sidebarOpen ? '260px' : '0px'} 1fr ${rightPanelOpen ? '320px' : '0px'}`;

  return (
    <div className="app-container" style={{ gridTemplateColumns: gridCols }}>
      {/* ── Header ──────────────────────── */}
      <Header
        stats={stats}
        currentZoom={currentZoom}
        onZoom={handleZoom}
        activeEdgeTypes={activeEdgeTypes}
        onEdgeToggle={handleEdgeToggle}
      />

      {/* ── Sidebar toggle ─────────────── */}
      {!sidebarOpen && (
        <button className="panel-collapse-btn" style={{ top: 70, left: 6 }} onClick={() => setSidebarOpen(true)}>
          ▶
        </button>
      )}
      {sidebarOpen && (
        <button className="panel-collapse-btn" style={{ top: 70, left: 260 }} onClick={() => setSidebarOpen(false)}>
          ◀
        </button>
      )}

      {/* ── Left Sidebar ───────────────── */}
      {sidebarOpen && (
        <Sidebar graph={store.graph} selectedNode={selectedNode} onNodeClick={handleNodeClick} />
      )}

      {/* ── WebGL Graph Canvas ─────────── */}
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 0 }}>
        <GraphCanvas
          graph={store.graph}
          viewState={store.viewState}
          selectedNode={selectedNode}
          onNodeClick={handleNodeClick}
          onNodeDoubleClick={handleNodeDoubleClick}
          nodeGlow={nodeGlow}
          edgeGlow={edgeGlow}
        />
        <Legend viewState={store.viewState} />
      </div>

      {/* ── Right panel toggle ─────────── */}
      {!rightPanelOpen && (
        <button className="panel-collapse-btn" style={{ top: 70, right: 6 }} onClick={() => setRightPanelOpen(true)}>
          ◀
        </button>
      )}
      {rightPanelOpen && (
        <button
          className="panel-collapse-btn"
          style={{ top: 70, right: 320 }}
          onClick={() => setRightPanelOpen(false)}
        >
          ▶
        </button>
      )}

      {/* ── Right Panel ────────────────── */}
      {rightPanelOpen && (
        <div className="sidebar-right">
          <div className="tab-bar">
            {(['inspect', 'chat', 'settings'] as const).map((tab) => (
              <button
                key={tab}
                className={`tab-btn ${rightTab === tab ? 'active' : ''}`}
                onClick={() => setRightTab(tab)}
              >
                {tab}
              </button>
            ))}
          </div>

          {rightTab === 'inspect' && (
            <InspectPanel
              nodeAttrs={selectedNodeData}
              graph={store.graph}
              selectedNode={selectedNode}
              onNodeClick={handleNodeClick}
            />
          )}

          {rightTab === 'chat' && (
            <ChatPanel
              messages={chatMessages}
              loading={chatLoading}
              onKeyDown={handleChatKeyDown}
              inputRef={chatInputRef}
              scrollRef={chatScrollRef}
            />
          )}

          {rightTab === 'settings' && (
            <SettingsPanel
              hasApiKey={hasApiKey}
              llmModel={llmModel}
              onSave={handleSaveSettings}
              nodeGlow={nodeGlow}
              edgeGlow={edgeGlow}
              onNodeGlowChange={setNodeGlow}
              onEdgeGlowChange={setEdgeGlow}
            />
          )}

          <div className="command-bar">
            <div className="command-input-wrapper">
              <span className="command-prefix">&gt;_</span>
              <input
                ref={commandRef}
                className="command-input"
                placeholder="view.isolate('analyzer') or natural language..."
                onKeyDown={handleKeyDown}
              />
            </div>
            {hasActiveFilters && (
              <button
                className="clear-filters-btn"
                onClick={() => sendCommand('view.reset_filter()')}
                title="Clear all filters and highlights"
              >
                ✕ CLEAR FILTERS
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── Status Bar ─────────────────── */}
      <StatusBar message={statusMsg} viewState={store.viewState} />
    </div>
  );
}

// ── Header ──────────────────────────────────────────────────────
function Header({
  stats,
  currentZoom,
  onZoom,
  activeEdgeTypes,
  onEdgeToggle,
}: {
  stats: Stats | null;
  currentZoom: string;
  onZoom: (level: string) => void;
  activeEdgeTypes: string[];
  onEdgeToggle: (type: string) => void;
}) {
  const edgeSet = new Set(activeEdgeTypes);
  const EDGE_CHIPS = [
    { type: 'calls', color: '#88bbdd' },
    { type: 'imports', color: '#667799' },
    { type: 'inherits', color: '#bb77ee' },
    { type: 'reads', color: '#33ccaa' },
    { type: 'writes', color: '#ff5566' },
    { type: 'returns', color: '#eebb44' },
    { type: 'contains', color: '#556677' },
  ];

  return (
    <div className="header">
      <div>
        <div className="logo">INTERLINKED</div>
        <div className="logo-sub">TOPOLOGY EXPLORER</div>
      </div>
      <span className="separator" />
      <div className="stats">
        {stats && (
          <>
            <span>MOD<span className="stat-value">{stats.modules}</span></span>
            <span>CLS<span className="stat-value">{stats.classes}</span></span>
            <span>FN<span className="stat-value">{stats.functions + stats.methods}</span></span>
            <span>VAR<span className="stat-value">{stats.variables}</span></span>
            <span>DEAD<span className="stat-value stat-dead">{stats.dead_nodes}</span></span>
            <span>EDGE<span className="stat-value">{stats.total_edges}</span></span>
          </>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginLeft: 'auto' }}>
        <div className="zoom-controls">
          {['all', 'module', 'class', 'function', 'variable'].map((level) => (
            <button
              key={level}
              className={`zoom-btn ${currentZoom === level ? 'active' : ''}`}
              onClick={() => onZoom(level)}
            >
              {level}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 3, justifyContent: 'flex-end', alignItems: 'center' }}>
          {EDGE_CHIPS.map(({ type, color }) => (
            <button
              key={type}
              className={`edge-chip ${edgeSet.has(type) ? 'active' : ''}`}
              style={edgeSet.has(type) ? { borderColor: color + '88', color } : {}}
              onClick={() => onEdgeToggle(type)}
            >
              {type}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Sidebar ─────────────────────────────────────────────────────
function Sidebar({
  graph,
  selectedNode,
  onNodeClick,
}: {
  graph: Graph;
  selectedNode: string | null;
  onNodeClick: (id: string) => void;
}) {
  const grouped = useMemo(() => {
    const groups: Record<string, { id: string; name: string; symbolType: string; isDead: boolean; isProposed: boolean }[]> = {};
    graph.forEachNode((id: string, attrs: Record<string, unknown>) => {
      const t = attrs.symbolType as string;
      if (!groups[t]) groups[t] = [];
      groups[t].push({
        id,
        name: attrs.label as string,
        symbolType: t,
        isDead: attrs.isDead as boolean,
        isProposed: attrs.isProposed as boolean,
      });
    });
    return groups;
  }, [graph, graph.order]);

  return (
    <div className="sidebar-left">
      {['module', 'class', 'function', 'method', 'variable'].map(
        (type) =>
          grouped[type] &&
          grouped[type].length > 0 && (
            <div className="sidebar-section" key={type}>
              <div className="sidebar-section-title">
                {type}s ({grouped[type].length})
              </div>
              {grouped[type].slice(0, 100).map((node) => (
                <div
                  key={node.id}
                  className={`node-list-item ${selectedNode === node.id ? 'active' : ''}`}
                  onClick={() => onNodeClick(node.id)}
                  title={node.id}
                >
                  <span className={`node-badge ${node.symbolType}`}>{node.symbolType.slice(0, 3)}</span>
                  <span className="node-name">{node.name}</span>
                  {node.isDead && <span className="dead-indicator" />}
                  {node.isProposed && <span className="proposed-indicator" />}
                </div>
              ))}
            </div>
          ),
      )}
    </div>
  );
}

// ── Inspect Panel ───────────────────────────────────────────────
function InspectPanel({
  nodeAttrs,
  graph,
  selectedNode,
  onNodeClick,
}: {
  nodeAttrs: Record<string, unknown> | null;
  graph: Graph;
  selectedNode: string | null;
  onNodeClick: (id: string) => void;
}) {
  const emptyQuote = useMemo(() => GOSLING_QUOTES[Math.floor(Math.random() * GOSLING_QUOTES.length)], []);

  if (!nodeAttrs || !selectedNode) {
    return (
      <div className="detail-panel">
        <div className="empty-state">
          <div className="gosling-quote" style={{ whiteSpace: 'pre-line' }}>{emptyQuote[0]}</div>
          <div className="gosling-sub">{emptyQuote[1]}</div>
          <div className="gosling-sub" style={{ marginTop: 12, color: '#2a4a6a', fontSize: 9 }}>
            SELECT A NODE TO INSPECT
          </div>
        </div>
      </div>
    );
  }

  const incoming: { id: string; label: string; edgeType: string }[] = [];
  const outgoing: { id: string; label: string; edgeType: string }[] = [];

  if (graph.hasNode(selectedNode)) {
    graph.forEachInEdge(selectedNode, (_edge: string, attrs: Record<string, unknown>, source: string) => {
      incoming.push({
        id: source,
        label: (graph.getNodeAttribute(source, 'label') as string) || source,
        edgeType: attrs.edgeType as string,
      });
    });
    graph.forEachOutEdge(selectedNode, (_edge: string, attrs: Record<string, unknown>, _src: string, target: string) => {
      outgoing.push({
        id: target,
        label: (graph.getNodeAttribute(target, 'label') as string) || target,
        edgeType: attrs.edgeType as string,
      });
    });
  }

  return (
    <div className="detail-panel">
      <div className="detail-title">{nodeAttrs.label as string}</div>

      <div className="detail-field">
        <div className="detail-label">QUALIFIED NAME</div>
        <div className="detail-value">
          <code>{nodeAttrs.qualifiedName as string}</code>
        </div>
      </div>

      <div className="detail-field">
        <div className="detail-label">TYPE</div>
        <div className="detail-value">{String(nodeAttrs.symbolType)}</div>
      </div>

      {!!nodeAttrs.filePath && (
        <div className="detail-field">
          <div className="detail-label">FILE</div>
          <div className="detail-value">
            {String(nodeAttrs.filePath)}:{String(nodeAttrs.lineStart)}
          </div>
        </div>
      )}

      {!!nodeAttrs.signature && (
        <div className="detail-field">
          <div className="detail-label">SIGNATURE</div>
          <div className="detail-value">
            <code>{String(nodeAttrs.signature)}</code>
          </div>
        </div>
      )}

      {!!nodeAttrs.docstring && (
        <div className="detail-field">
          <div className="detail-label">DOCSTRING</div>
          <div className="detail-value" style={{ whiteSpace: 'pre-wrap', fontSize: 10 }}>
            {String(nodeAttrs.docstring)}
          </div>
        </div>
      )}

      {incoming.length > 0 && (
        <div className="detail-field">
          <div className="detail-label">INCOMING ({incoming.length})</div>
          <ul style={{ listStyle: 'none', marginTop: 4 }}>
            {incoming.slice(0, 30).map((e, i) => (
              <li
                key={i}
                style={{ fontSize: 10, padding: '3px 0', color: '#5a8aaa', cursor: 'pointer' }}
                onClick={() => onNodeClick(e.id)}
              >
                <span
                  style={{
                    fontFamily: "'Orbitron', sans-serif",
                    fontSize: 7,
                    padding: '1px 5px',
                    borderRadius: 2,
                    marginRight: 6,
                    fontWeight: 600,
                    letterSpacing: 1,
                    border: `1px solid ${EDGE_COLORS[e.edgeType] || '#334'}`,
                    color: EDGE_COLORS[e.edgeType] || '#556',
                  }}
                >
                  {e.edgeType}
                </span>
                {e.label}
              </li>
            ))}
          </ul>
        </div>
      )}

      {outgoing.length > 0 && (
        <div className="detail-field">
          <div className="detail-label">OUTGOING ({outgoing.length})</div>
          <ul style={{ listStyle: 'none', marginTop: 4 }}>
            {outgoing.slice(0, 30).map((e, i) => (
              <li
                key={i}
                style={{ fontSize: 10, padding: '3px 0', color: '#5a8aaa', cursor: 'pointer' }}
                onClick={() => onNodeClick(e.id)}
              >
                <span
                  style={{
                    fontFamily: "'Orbitron', sans-serif",
                    fontSize: 7,
                    padding: '1px 5px',
                    borderRadius: 2,
                    marginRight: 6,
                    fontWeight: 600,
                    letterSpacing: 1,
                    border: `1px solid ${EDGE_COLORS[e.edgeType] || '#334'}`,
                    color: EDGE_COLORS[e.edgeType] || '#556',
                  }}
                >
                  {e.edgeType}
                </span>
                {e.label}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Chat Panel ──────────────────────────────────────────────────
function ChatPanel({
  messages,
  loading,
  onKeyDown,
  inputRef,
  scrollRef,
}: {
  messages: { role: string; content: string; commands?: string[] }[];
  loading: boolean;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  inputRef: React.RefObject<HTMLTextAreaElement>;
  scrollRef: React.RefObject<HTMLDivElement>;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      <div className="chat-messages" ref={scrollRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            {msg.content}
            {msg.commands && msg.commands.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 9, color: '#4a7a9a' }}>
                {msg.commands.map((cmd, j) => (
                  <div key={j} style={{ fontFamily: "'Share Tech Mono', monospace" }}>
                    → {cmd}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg system" style={{ animation: 'pulse-glow 1s infinite' }}>
            PROCESSING...
          </div>
        )}
      </div>
      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          className="chat-input"
          rows={2}
          placeholder="Ask about the codebase..."
          onKeyDown={onKeyDown}
        />
      </div>
    </div>
  );
}

// ── Settings Panel ──────────────────────────────────────────────
function SettingsPanel({
  hasApiKey,
  llmModel,
  onSave,
  nodeGlow,
  edgeGlow,
  onNodeGlowChange,
  onEdgeGlowChange,
}: {
  hasApiKey: boolean;
  llmModel: string;
  onSave: (apiKey: string, model: string) => void;
  nodeGlow: number;
  edgeGlow: number;
  onNodeGlowChange: (v: number) => void;
  onEdgeGlowChange: (v: number) => void;
}) {
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState(llmModel);

  const sliderStyle = { width: '100%', accentColor: '#00d4ff', cursor: 'pointer' };

  return (
    <div className="detail-panel">
      <div className="detail-title">SETTINGS</div>

      <div className="detail-field">
        <div className="detail-label">GLOBAL GLOW <span style={{ color: '#00d4ff' }}>{Math.round(((nodeGlow + edgeGlow) / 2) * 100)}%</span></div>
        <input
          type="range" min={0} max={1} step={0.05}
          value={(nodeGlow + edgeGlow) / 2}
          onChange={(e) => { const v = Number(e.target.value); onNodeGlowChange(v); onEdgeGlowChange(v); }}
          style={sliderStyle}
        />
      </div>

      <div className="detail-field">
        <div className="detail-label">NODE GLOW <span style={{ color: '#0099dd' }}>{Math.round(nodeGlow * 100)}%</span></div>
        <input
          type="range" min={0} max={1} step={0.05}
          value={nodeGlow}
          onChange={(e) => onNodeGlowChange(Number(e.target.value))}
          style={{ ...sliderStyle, accentColor: '#0099dd' }}
        />
      </div>

      <div className="detail-field">
        <div className="detail-label">EDGE GLOW <span style={{ color: '#88bbdd' }}>{Math.round(edgeGlow * 100)}%</span></div>
        <input
          type="range" min={0} max={1} step={0.05}
          value={edgeGlow}
          onChange={(e) => onEdgeGlowChange(Number(e.target.value))}
          style={{ ...sliderStyle, accentColor: '#88bbdd' }}
        />
      </div>

      <div className="detail-field">
        <div className="detail-label">API KEY STATUS</div>
        <div className="detail-value">
          {hasApiKey ? (
            <span style={{ color: '#39ff14', textShadow: '0 0 6px #39ff1444' }}>CONFIGURED</span>
          ) : (
            <span style={{ color: '#ff2975', textShadow: '0 0 6px #ff297544' }}>NOT SET</span>
          )}
        </div>
      </div>

      <div className="detail-field">
        <div className="detail-label">ANTHROPIC API KEY</div>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-ant-..."
          style={{
            width: '100%',
            background: '#000',
            border: '1px solid #1a3a5c',
            borderRadius: 2,
            color: '#c0dff0',
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 11,
            padding: '6px 8px',
            outline: 'none',
          }}
        />
      </div>

      <div className="detail-field">
        <div className="detail-label">MODEL</div>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{
            width: '100%',
            background: '#000',
            border: '1px solid #1a3a5c',
            borderRadius: 2,
            color: '#c0dff0',
            fontFamily: "'Share Tech Mono', monospace",
            fontSize: 11,
            padding: '6px 8px',
            outline: 'none',
          }}
        />
      </div>

      <button
        onClick={() => onSave(apiKey, model)}
        style={{
          padding: '8px 20px',
          background: 'transparent',
          border: '1px solid #00d4ff55',
          color: '#00d4ff',
          fontFamily: "'Orbitron', sans-serif",
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: 2,
          cursor: 'pointer',
          borderRadius: 2,
          textTransform: 'uppercase' as const,
          textShadow: '0 0 6px #00d4ff44',
          marginTop: 8,
        }}
      >
        SAVE
      </button>
    </div>
  );
}

// ── Legend ──────────────────────────────────────────────────────
function Legend({ viewState }: { viewState: ViewState | null }) {
  const [collapsed, setCollapsed] = useState(false);
  const hasTrace =
    (viewState?.trace_node_roles && Object.keys(viewState.trace_node_roles).length > 0) ||
    (viewState?.trace_edge_roles && Object.keys(viewState.trace_edge_roles).length > 0);

  const shapeIcon = (type: string, color: string, sz = 14) => {
    const r = sz / 2;
    const shapeFn = NODE_SHAPES[type] || NODE_SHAPES.function;
    const d = shapeFn(r);
    return (
      <svg width={sz + 2} height={sz + 2} style={{ flexShrink: 0, verticalAlign: 'middle' }}>
        <g transform={`translate(${r + 1},${r + 1})`}>
          <path d={d} fill={color} stroke={color} strokeWidth="0.5" style={{ filter: `drop-shadow(0 0 3px ${color})` }} />
        </g>
      </svg>
    );
  };

  const lineIcon = (color: string, dash: string | null | undefined, w = 1.5) => (
    <svg width={30} height={6} style={{ flexShrink: 0, verticalAlign: 'middle' }}>
      <line x1="0" y1="3" x2="30" y2="3" stroke={color} strokeWidth={Math.max(w, 1.5)}
        strokeDasharray={dash || undefined} style={{ filter: `drop-shadow(0 0 2px ${color})` }} />
    </svg>
  );

  const dot = (color: string) => (
    <svg width={10} height={10} style={{ flexShrink: 0, verticalAlign: 'middle' }}>
      <circle cx="5" cy="5" r="4" fill={color} style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
    </svg>
  );

  if (collapsed) {
    return (
      <div className="legend-bar collapsed" onClick={() => setCollapsed(false)}>
        <span style={{ fontFamily: "'Orbitron', sans-serif", fontSize: 9, letterSpacing: 2, color: 'var(--text-muted)' }}>
          ▲ LEGEND
        </span>
      </div>
    );
  }

  return (
    <div className="legend-bar" style={{ position: 'relative' }}>
      <button className="legend-collapse-btn" onClick={() => setCollapsed(true)}>▼</button>

      <div className="legend-group">
        <div className="legend-group-title">SYMBOLS</div>
        {Object.entries(NODE_COLORS).map(([t, c]) => (
          <div className="legend-row" key={t}>{shapeIcon(t, c)} <span style={{ color: c }}>{t}</span></div>
        ))}
      </div>

      <div className="legend-group">
        <div className="legend-group-title">RELATIONSHIPS</div>
        {Object.entries(EDGE_COLORS).filter(([t]) => t !== 'contains' && t !== 'proposed').map(([t, c]) => (
          <div className="legend-row" key={t}>{lineIcon(c, EDGE_DASH[t], EDGE_WIDTH[t])} <span style={{ color: c }}>{t}</span></div>
        ))}
      </div>

      {hasTrace && (
        <>
          <div className="legend-group">
            <div className="legend-group-title">TRACE ROLES</div>
            {Object.entries(TRACE_ROLE_COLORS).map(([r, c]) => (
              <div className="legend-row" key={`tr-${r}`}>{dot(c)} <span style={{ color: c }}>{r}</span></div>
            ))}
          </div>
          <div className="legend-group">
            <div className="legend-group-title">TRACE EDGES</div>
            {Object.entries(TRACE_EDGE_COLORS).map(([r, c]) => (
              <div className="legend-row" key={`te-${r}`}>{lineIcon(c, null, 2.5)} <span style={{ color: c }}>{r}</span></div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// ── Status Bar ──────────────────────────────────────────────────
function StatusBar({ message, viewState }: { message: string; viewState: ViewState | null }) {
  const [radioPlaying, setRadioPlaying] = useState(false);
  const [stationIdx, setStationIdx] = useState(0);
  const [volume, setVolume] = useState(0.3);
  const [nowPlaying, setNowPlaying] = useState('');
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const npIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const station = RADIO_STATIONS[stationIdx];

  // Fetch now-playing metadata from station API
  const fetchNowPlaying = useCallback(() => {
    const st = RADIO_STATIONS[stationIdx];
    if (!st.nowPlayingUrl) {
      setNowPlaying(st.name + '  ///  STREAMING LIVE');
      return;
    }
    fetch(st.nowPlayingUrl)
      .then((r) => r.json())
      .then((data) => {
        let text = '';
        if (st.parser === 'somafm') {
          const song = data.songs?.[0];
          if (song) text = `${song.artist}  —  ${song.title}`;
        } else if (st.parser === 'plaza') {
          const song = data.song;
          if (song) text = `${song.artist}  —  ${song.title}`;
        }
        setNowPlaying(text || st.name);
      })
      .catch(() => setNowPlaying(st.name));
  }, [stationIdx]);

  // Poll now-playing every 15s when playing
  useEffect(() => {
    if (radioPlaying) {
      fetchNowPlaying();
      npIntervalRef.current = setInterval(fetchNowPlaying, 15000);
    } else {
      setNowPlaying('');
    }
    return () => {
      if (npIntervalRef.current) clearInterval(npIntervalRef.current);
    };
  }, [radioPlaying, fetchNowPlaying]);

  const toggleRadio = useCallback(() => {
    if (radioPlaying) {
      audioRef.current?.pause();
      audioRef.current!.src = '';
      audioRef.current = null;
      setRadioPlaying(false);
      setNowPlaying('');
    } else {
      audioRef.current = new Audio(station.url);
      audioRef.current.volume = volume;
      audioRef.current.crossOrigin = 'anonymous';
      audioRef.current.play().catch(() => {});
      setRadioPlaying(true);
    }
  }, [radioPlaying, station, volume]);

  const changeStation = useCallback(
    (idx: number) => {
      const wasPlaying = radioPlaying;
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
        audioRef.current = null;
        setRadioPlaying(false);
      }
      setStationIdx(idx);
      setNowPlaying('');
      if (wasPlaying) {
        setTimeout(() => {
          const st = RADIO_STATIONS[idx];
          audioRef.current = new Audio(st.url);
          audioRef.current.volume = volume;
          audioRef.current.crossOrigin = 'anonymous';
          audioRef.current.play().catch(() => {});
          setRadioPlaying(true);
        }, 100);
      }
    },
    [volume, radioPlaying],
  );

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  return (
    <div className="status-bar" style={{ display: 'flex', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: '1 1 0', minWidth: 0 }}>
        <span className="status-dot" />
        <span className="status-msg">{message}</span>
      </div>

      {/* Radio — centered */}
      <div className="radio-container" style={{ flex: '0 0 auto' }}>
        <select
          className="radio-station-select"
          value={stationIdx}
          onChange={(e) => changeStation(Number(e.target.value))}
        >
          {RADIO_STATIONS.map((s, i) => (
            <option key={s.id} value={i}>
              {s.name}
            </option>
          ))}
        </select>
        <button className={`radio-play-btn ${radioPlaying ? 'playing' : ''}`} onClick={toggleRadio}>
          {radioPlaying ? '■' : '▶'}
        </button>
        <input
          type="range"
          className="radio-volume"
          min={0}
          max={1}
          step={0.05}
          value={volume}
          onChange={(e) => setVolume(Number(e.target.value))}
        />
        {nowPlaying && (
          <div className="radio-nowplaying">
            <span className="radio-nowplaying-text">{nowPlaying}</span>
          </div>
        )}
      </div>

      <div style={{ flex: '1 1 0', display: 'flex', justifyContent: 'flex-end', minWidth: 0 }}>
        {viewState && (
          <span style={{ color: '#2a4a6a', whiteSpace: 'nowrap' }}>
            ZOOM:{viewState.zoom_level.toUpperCase()} // VIEW:{viewState.visible_node_ids.length || 'ALL'}
          </span>
        )}
      </div>
    </div>
  );
}
