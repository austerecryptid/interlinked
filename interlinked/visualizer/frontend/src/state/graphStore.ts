// ── Graph Store — Graphology instance + delta application ───────
// This is the single source of truth for the client-side graph.
// Backend pushes snapshots/deltas via SSE; this module applies them.

import Graph from 'graphology';
import { NODE_COLORS, NODE_SIZES, EDGE_COLORS, TRACE_ROLE_COLORS, TRACE_EDGE_COLORS } from '../theme';
import type { NodeData, EdgeData, ViewState, GraphSnapshot, GraphDelta } from '../types';

export interface GraphStore {
  graph: Graph;
  viewState: ViewState | null;
  layout: Record<string, { x: number; y: number }>;
  /** Synchronous selected-node tracker — Sigma reads this, not React state.
   *  Cleared before graph mutations so Sigma never sees a stale ID. */
  selectedNodeId: string | null;
}

export function createGraphStore(): GraphStore {
  return {
    graph: new Graph({ multi: true, type: 'directed', allowSelfLoops: true }),
    viewState: null,
    layout: {},
    selectedNodeId: null,
  };
}

function nodeAttrs(node: NodeData, viewState: ViewState | null, layout: Record<string, { x: number; y: number }>) {
  const pos = layout[node.id];
  const highlighted = viewState?.highlighted_node_ids?.includes(node.id) ?? false;
  const hasHighlights = (viewState?.highlighted_node_ids?.length ?? 0) > 0;
  const dimmed = hasHighlights && !highlighted;
  const traceRole = viewState?.trace_node_roles?.[node.id];

  // Determine color: trace role > highlight > normal > dimmed
  let color = NODE_COLORS[node.symbol_type] || '#55aacc';
  if (traceRole && TRACE_ROLE_COLORS[traceRole]) {
    color = TRACE_ROLE_COLORS[traceRole];
  }

  // Size
  const baseSize = NODE_SIZES[node.symbol_type] || 8;
  const size = highlighted ? baseSize * 1.4 : dimmed ? baseSize * 0.7 : baseSize;

  return {
    x: pos?.x ?? Math.random() * 1000,
    y: pos?.y ?? Math.random() * 800,
    size,
    color,
    label: node.name,
    // Custom attributes for rendering
    symbolType: node.symbol_type,
    qualifiedName: node.qualified_name,
    isDead: node.is_dead,
    isProposed: node.is_proposed,
    highlighted,
    dimmed,
    traceRole: traceRole || null,
    filePath: node.file_path,
    lineStart: node.line_start,
    lineEnd: node.line_end,
    docstring: node.docstring,
    signature: node.signature,
  };
}

function edgeAttrs(edge: EdgeData, viewState: ViewState | null) {
  const highlightedSet = new Set(viewState?.highlighted_node_ids ?? []);
  const hasHighlights = highlightedSet.size > 0;
  const srcHighlighted = highlightedSet.has(edge.source);
  const tgtHighlighted = highlightedSet.has(edge.target);
  const bothHighlighted = srcHighlighted && tgtHighlighted;
  const eitherHighlighted = srcHighlighted || tgtHighlighted;

  const traceKey = `${edge.source}|${edge.target}`;
  const traceRole = viewState?.trace_edge_roles?.[traceKey];

  let color = EDGE_COLORS[edge.edge_type] || '#1a3a5c';
  if (traceRole && TRACE_EDGE_COLORS[traceRole]) {
    color = TRACE_EDGE_COLORS[traceRole];
  }

  // During a trace/filter, dim edges unless BOTH endpoints are in the highlighted set.
  // This prevents bright edges from traced nodes going to non-traced nodes.
  const dimmed = hasHighlights && !bothHighlighted;

  return {
    color,
    size: bothHighlighted ? 4 : dimmed ? 0.5 : 2.5,
    edgeType: edge.edge_type,
    isDead: edge.is_dead,
    isProposed: edge.is_proposed,
    highlighted: bothHighlighted,
    dimmed,
    traceRole: traceRole || null,
  };
}

export function applySnapshot(store: GraphStore, snapshot: GraphSnapshot): void {
  const { graph } = store;

  // Clear stale selection BEFORE graph.clear() fires graphology events → Sigma render
  const newNodeIds = new Set(snapshot.nodes.map(n => n.id));
  if (store.selectedNodeId && !newNodeIds.has(store.selectedNodeId)) {
    store.selectedNodeId = null;
  }

  graph.clear();

  store.viewState = snapshot.view;
  store.layout = snapshot.layout || {};

  // Add nodes
  for (const node of snapshot.nodes) {
    graph.addNode(node.id, nodeAttrs(node, snapshot.view, store.layout));
  }

  // Add edges (skip contains at coarse zoom — they're implicit)
  for (const edge of snapshot.edges) {
    if (edge.edge_type === 'contains') continue;
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
    const key = `${edge.source}→${edge.target}:${edge.edge_type}`;
    if (!graph.hasEdge(key)) {
      graph.addEdgeWithKey(key, edge.source, edge.target, edgeAttrs(edge, snapshot.view));
    }
  }
}

export function applyDelta(store: GraphStore, delta: GraphDelta): void {
  const { graph } = store;

  // Clear stale selection BEFORE any mutations fire graphology events → Sigma render
  const removedSet = new Set(delta.removed_node_ids);
  if (store.selectedNodeId && removedSet.has(store.selectedNodeId)) {
    store.selectedNodeId = null;
  }

  store.viewState = delta.view;

  // Apply layout updates
  if (delta.layout_updates) {
    for (const [nid, pos] of Object.entries(delta.layout_updates)) {
      store.layout[nid] = pos;
    }
  }

  // Remove edges FIRST so Sigma never sees edges pointing to deleted nodes
  for (const edge of delta.removed_edges) {
    const key = `${edge.source}→${edge.target}:${edge.edge_type}`;
    if (graph.hasEdge(key)) graph.dropEdge(key);
  }

  // Remove nodes (cascades remaining edges in graphology)
  for (const id of delta.removed_node_ids) {
    if (graph.hasNode(id)) graph.dropNode(id);
    delete store.layout[id];
  }

  // Add new nodes
  for (const node of delta.added_nodes) {
    if (!graph.hasNode(node.id)) {
      graph.addNode(node.id, nodeAttrs(node, delta.view, store.layout));
    }
  }

  // Update existing nodes (highlight state, dead status changes, etc.)
  for (const node of delta.updated_nodes) {
    if (graph.hasNode(node.id)) {
      graph.mergeNodeAttributes(node.id, nodeAttrs(node, delta.view, store.layout));
    }
  }

  // Add edges
  for (const edge of delta.added_edges) {
    if (edge.edge_type === 'contains') continue;
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
    const key = `${edge.source}→${edge.target}:${edge.edge_type}`;
    if (!graph.hasEdge(key)) {
      graph.addEdgeWithKey(key, edge.source, edge.target, edgeAttrs(edge, delta.view));
    }
  }

  // Safety: drop any orphaned edges that survived the delta
  const edgesToDrop: string[] = [];
  graph.forEachEdge((key, _attrs, source, target) => {
    if (!graph.hasNode(source) || !graph.hasNode(target)) edgesToDrop.push(key);
  });
  for (const key of edgesToDrop) graph.dropEdge(key);

  // Re-apply view state to all existing nodes (highlights may have changed)
  graph.forEachNode((nodeId: string, attrs: Record<string, unknown>) => {
    const highlighted = delta.view.highlighted_node_ids?.includes(nodeId) ?? false;
    const hasHighlights = (delta.view.highlighted_node_ids?.length ?? 0) > 0;
    const dimmed = hasHighlights && !highlighted;
    const traceRole = delta.view.trace_node_roles?.[nodeId];

    let color = NODE_COLORS[attrs.symbolType as string] || '#55aacc';
    if (traceRole && TRACE_ROLE_COLORS[traceRole]) {
      color = TRACE_ROLE_COLORS[traceRole];
    }

    const baseSize = NODE_SIZES[attrs.symbolType as string] || 8;
    const size = highlighted ? baseSize * 1.4 : dimmed ? baseSize * 0.7 : baseSize;

    graph.mergeNodeAttributes(nodeId, {
      color,
      size,
      highlighted,
      dimmed,
      traceRole: traceRole || null,
    });
  });
}
