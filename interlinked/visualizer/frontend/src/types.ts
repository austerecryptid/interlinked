// ── Shared types mirroring backend Pydantic models ──────────────

export interface NodeData {
  id: string;
  name: string;
  qualified_name: string;
  symbol_type: string;
  file_path: string | null;
  line_start: number | null;
  line_end: number | null;
  docstring: string | null;
  signature: string | null;
  is_dead: boolean;
  is_proposed: boolean;
  metadata: Record<string, unknown>;
}

export interface EdgeData {
  source: string;
  target: string;
  edge_type: string;
  is_dead: boolean;
  is_proposed: boolean;
  line: number | null;
  metadata: Record<string, unknown>;
}

export interface ViewContext {
  what: string;
  why: string;
  where: string;
  source: string;
}

export interface ViewState {
  zoom_level: string;
  focus_node: string | null;
  focus_depth: number;
  visible_node_ids: string[];
  visible_edge_types: string[];
  highlighted_node_ids: string[];
  highlighted_edge_ids: [string, string][];
  trace_node_roles: Record<string, string>;
  trace_edge_roles: Record<string, string>;
  show_dead: boolean;
  show_proposed: boolean;
  filter_expression: string | null;
  context: ViewContext;
}

export interface GraphSnapshot {
  nodes: NodeData[];
  edges: EdgeData[];
  view: ViewState;
  layout: Record<string, { x: number; y: number }>;
}

export interface GraphDelta {
  added_nodes: NodeData[];
  removed_node_ids: string[];
  updated_nodes: NodeData[];
  added_edges: EdgeData[];
  removed_edges: EdgeData[];
  view: ViewState;
  layout_updates: Record<string, { x: number; y: number }>;
}

export interface SSEMessage {
  type: 'snapshot' | 'delta';
  data: GraphSnapshot | GraphDelta;
}

export interface Stats {
  modules: number;
  classes: number;
  functions: number;
  methods: number;
  variables: number;
  parameters: number;
  external_calls: number;
  dead_nodes: number;
  total_edges: number;
}
