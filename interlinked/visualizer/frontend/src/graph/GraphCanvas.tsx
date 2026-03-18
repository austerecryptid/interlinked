// ── WebGL Graph Canvas — Sigma.js v3 + Graphology ───────────────
// GPU-accelerated rendering. Glow is free here — no SVG filter perf tax.
// ForceAtlas2 for fast force-directed layout on the client.

import React, { useEffect, useRef, useCallback } from 'react';
import Sigma from 'sigma';
import Graph from 'graphology';
import { createEdgeArrowProgram } from 'sigma/rendering';
import FA2Layout from 'graphology-layout-forceatlas2/worker';
import { inferSettings } from 'graphology-layout-forceatlas2';
import { GOSLING_QUOTES, NODE_COLORS, NODE_SIZES } from '../theme';
import { NODE_PROGRAM_CLASSES, SYMBOL_TYPE_MAP } from './nodePrograms';
import type { ViewState, NodeData } from '../types';

// Arrow program with large, visible arrowheads
const BigArrowProgram = createEdgeArrowProgram({
  lengthToThicknessRatio: 3.5,
  widenessToThicknessRatio: 4,
});

interface GraphCanvasProps {
  graph: Graph;
  viewState: ViewState | null;
  selectedNode: string | null;
  onNodeClick: (nodeId: string) => void;
  onNodeDoubleClick: (nodeId: string) => void;
  nodeGlow: number;
  edgeGlow: number;
}

export default function GraphCanvas({
  graph,
  viewState,
  selectedNode,
  onNodeClick,
  onNodeDoubleClick,
  nodeGlow,
  edgeGlow,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const nodeBloomRef = useRef<HTMLCanvasElement | null>(null);
  const edgeBloomRef = useRef<HTMLCanvasElement | null>(null);
  const fa2Ref = useRef<FA2Layout | null>(null);
  const fa2TimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync ref for Sigma — Sigma's nodeReducer reads this synchronously.
  // React state is async; the ref prevents Sigma from seeing a stale selectedNode
  // after graph mutations remove the node.
  const selectedNodeRef = useRef<string | null>(selectedNode);
  selectedNodeRef.current = selectedNode;

  // ForceAtlas2 layout — run a burst when graph content changes.
  // If there are highlighted nodes (trace/query), pin everything else and
  // only lay out the active subset. On plain view switches, lay out everything.
  const highlightedIds = viewState?.highlighted_node_ids;
  useEffect(() => {
    if (graph.order === 0) return;

    // Stop any existing layout
    if (fa2Ref.current) {
      fa2Ref.current.stop();
      fa2Ref.current.kill();
      fa2Ref.current = null;
    }
    if (fa2TimerRef.current) {
      clearTimeout(fa2TimerRef.current);
    }

    try {
      const n = graph.order;
      const activeSet = new Set(highlightedIds ?? []);
      const hasHighlights = activeSet.size > 0;

      // Pin / unpin nodes: if we have highlighted nodes, only they move.
      // On a plain view switch (no highlights), everything is free.
      graph.forEachNode((id, attrs) => {
        attrs.fixed = hasHighlights && !activeSet.has(id) ? 1 : 0;
      });

      // How many nodes are actually free to move?
      const freeCount = hasHighlights ? activeSet.size : n;

      // Normalize positions of FREE nodes into a consistent range.
      // Use a wider spread for highlighted subgraphs so repulsion has room to work.
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      graph.forEachNode((id, attrs) => {
        if (attrs.fixed) return;
        if (typeof attrs.x === 'number' && typeof attrs.y === 'number') {
          if (attrs.x < minX) minX = attrs.x;
          if (attrs.x > maxX) maxX = attrs.x;
          if (attrs.y < minY) minY = attrs.y;
          if (attrs.y > maxY) maxY = attrs.y;
        }
      });
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const targetSpread = hasHighlights
        ? Math.max(freeCount * 300, 2000)   // wide spread for subgraphs
        : Math.max(freeCount * 80, 800);
      graph.forEachNode((id, attrs) => {
        if (attrs.fixed) return;
        if (typeof attrs.x === 'number') attrs.x = ((attrs.x - minX) / rangeX - 0.5) * targetSpread;
        if (typeof attrs.y === 'number') attrs.y = ((attrs.y - minY) / rangeY - 0.5) * targetSpread;
      });

      const settings = inferSettings(graph);

      // Highlighted subgraphs: very low gravity + high repulsion so nodes spread out
      // Full graph: moderate gravity to keep clusters together
      // inferSettings is calibrated for the full graph — override for subsets
      let gravity: number, scalingRatio: number, slowDown: number;
      if (hasHighlights) {
        // Near-zero gravity + extreme repulsion for readable subgraphs
        gravity = 0.0001;
        scalingRatio = freeCount > 100 ? 200 : freeCount > 20 ? 150 : 100;
        slowDown = freeCount > 100 ? 3 : 1;
      } else {
        gravity = freeCount > 500 ? 0.001 : freeCount > 100 ? 0.005 : 0.01;
        scalingRatio = freeCount > 500 ? 80 : freeCount > 100 ? 50 : freeCount > 20 ? 30 : 15;
        slowDown = freeCount > 500 ? 15 : freeCount > 100 ? 8 : 4;
      }

      const layout = new FA2Layout(graph, {
        settings: {
          ...settings,
          gravity,
          scalingRatio,
          slowDown,
          strongGravityMode: false,
          barnesHutOptimize: n > 50,
          adjustSizes: true,
        },
      });
      fa2Ref.current = layout;
      layout.start();

      // Highlighted subgraphs get longer duration so repulsion fully separates nodes
      const duration = hasHighlights
        ? (freeCount > 100 ? 2500 : freeCount > 20 ? 1800 : 1200)
        : (freeCount > 2000 ? 4000 : freeCount > 500 ? 2500 : freeCount > 100 ? 1500 : 800);
      fa2TimerRef.current = setTimeout(() => {
        layout.stop();

        // ── Post-layout: enforce minimum distance via spatial grid ────
        // Rebuild grid each iteration so pushed nodes get rechecked.
        // O(n) per iteration. Scales to 150k+ nodes.
        const MIN_DIST = hasHighlights ? 300 : 160;
        const MIN_DIST_SQ = MIN_DIST * MIN_DIST;

        // Collect positions into arrays for cache-friendly iteration
        const ids: string[] = [];
        const xs: number[] = [];
        const ys: number[] = [];
        graph.forEachNode((id, attrs) => {
          if (typeof attrs.x === 'number' && typeof attrs.y === 'number') {
            ids.push(id);
            xs.push(attrs.x);
            ys.push(attrs.y);
          }
        });
        const count = ids.length;

        const iterations = count > 10000 ? 2 : count > 1000 ? 4 : 8;
        for (let iter = 0; iter < iterations; iter++) {
          // Rebuild grid every iteration
          const cellSize = MIN_DIST;
          const grid = new Map<number, number[]>();
          const PRIME = 73856093;
          const PRIME2 = 19349663;
          for (let i = 0; i < count; i++) {
            const cx = Math.floor(xs[i] / cellSize);
            const cy = Math.floor(ys[i] / cellSize);
            const key = (cx * PRIME) ^ (cy * PRIME2);
            const cell = grid.get(key);
            if (cell) cell.push(i);
            else grid.set(key, [i]);
          }

          // For each node, check 9 neighboring cells
          for (let i = 0; i < count; i++) {
            const cx = Math.floor(xs[i] / cellSize);
            const cy = Math.floor(ys[i] / cellSize);
            for (let dcx = -1; dcx <= 1; dcx++) {
              for (let dcy = -1; dcy <= 1; dcy++) {
                const key = ((cx + dcx) * PRIME) ^ ((cy + dcy) * PRIME2);
                const cell = grid.get(key);
                if (!cell) continue;
                for (const j of cell) {
                  if (j <= i) continue;
                  let dx = xs[j] - xs[i];
                  let dy = ys[j] - ys[i];
                  const distSq = dx * dx + dy * dy;
                  if (distSq >= MIN_DIST_SQ) continue;
                  const dist = Math.sqrt(distSq);
                  const overlap = (MIN_DIST - dist) / 2;
                  if (dist < 0.1) {
                    dx = (Math.random() - 0.5); dy = (Math.random() - 0.5);
                    const jl = Math.sqrt(dx * dx + dy * dy) || 1;
                    dx /= jl; dy /= jl;
                  } else {
                    dx /= dist; dy /= dist;
                  }
                  xs[i] -= dx * overlap;
                  ys[i] -= dy * overlap;
                  xs[j] += dx * overlap;
                  ys[j] += dy * overlap;
                }
              }
            }
          }
        }

        // Write resolved positions back to the graph
        for (let i = 0; i < count; i++) {
          if (graph.hasNode(ids[i])) {
            graph.setNodeAttribute(ids[i], 'x', xs[i]);
            graph.setNodeAttribute(ids[i], 'y', ys[i]);
          }
        }

        // Clean up fixed flags so they don't leak into rendering
        graph.forEachNode((_id, attrs) => { attrs.fixed = 0; });
      }, duration);
    } catch {
      // FA2 can fail on empty/degenerate graphs
    }

    return () => {
      if (fa2Ref.current) {
        fa2Ref.current.stop();
        fa2Ref.current.kill();
        fa2Ref.current = null;
      }
      if (fa2TimerRef.current) {
        clearTimeout(fa2TimerRef.current);
      }
      // Clean up fixed flags on teardown
      graph.forEachNode((_id, attrs) => { attrs.fixed = 0; });
    };
  }, [graph, graph.order, highlightedIds]);

  // Initialize / re-initialize Sigma when graph gains nodes
  // We track nodeCount so this fires when data first arrives via SSE.
  const nodeCount = graph.order;

  useEffect(() => {
    if (!containerRef.current || nodeCount === 0) return;

    // Kill previous instance if any (e.g. graph was cleared and repopulated)
    if (sigmaRef.current) {
      sigmaRef.current.kill();
      sigmaRef.current = null;
    }

    // Temporarily force preserveDrawingBuffer so bloom canvas can read WebGL output.
    // Scoped tightly to Sigma construction only; restored immediately after.
    const _origGetCtx = HTMLCanvasElement.prototype.getContext;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (HTMLCanvasElement.prototype as any).getContext = function (this: HTMLCanvasElement, t: string, a?: any) {
      if (t === 'webgl' || t === 'webgl2') a = { ...a, preserveDrawingBuffer: true };
      return _origGetCtx.call(this, t, a);
    };
    const renderer = new Sigma(graph, containerRef.current, {
      allowInvalidContainer: true,
      renderLabels: true,
      renderEdgeLabels: false,
      labelFont: '"Share Tech Mono", "Courier New", monospace',
      labelSize: 10,
      labelColor: { color: '#c0dff0' },
      labelDensity: 0.5,
      labelGridCellSize: 100,
      labelRenderedSizeThreshold: 4,
      edgeLabelFont: '"Share Tech Mono", monospace',
      stagePadding: 40,
      // WebGL shape programs — each symbol type gets a distinct shape
      defaultNodeType: 'circle',
      nodeProgramClasses: NODE_PROGRAM_CLASSES as Record<string, any>,
      defaultEdgeType: 'arrow',
      edgeProgramClasses: {
        arrow: BigArrowProgram,
      },
      // Node reducers: set `type` for shape + dynamic styling
      nodeReducer: (node, data) => {
        const res = { ...data };
        // Route node to correct WebGL shape program
        const symbolType = (res as any).symbolType as string | undefined;
        res.type = SYMBOL_TYPE_MAP[symbolType || ''] || 'circle';
        if (res.highlighted) {
          res.size = (res.size || 8) * 1.2;
        }
        if (node === selectedNodeRef.current && graph.hasNode(node)) {
          res.highlighted = true;
          res.color = '#ffe600';
        }
        if (res.dimmed) {
          res.color = '#0c0c14';
          res.label = undefined;
        }
        return res;
      },
      // Edge reducers for dynamic styling
      edgeReducer: (_edge, data) => {
        const res = { ...data };
        if (res.dimmed) {
          res.color = '#080810';
          res.size = 0.2;
        }
        return res;
      },
      defaultNodeColor: '#55aacc',
      defaultEdgeColor: '#1a3a5c',
      // Disable white outlines + label backgrounds on highlighted nodes
      defaultDrawNodeHover: () => {},
      zIndex: true,
    });

    // Restore original getContext immediately
    HTMLCanvasElement.prototype.getContext = _origGetCtx;

    sigmaRef.current = renderer;

    // ── Bloom post-processing: two layers (nodes + edges) ─────────
    // Separate canvases so node glow and edge glow have independent opacity.
    // Nodes at 1/2 res (circles need more pixels to stay round).
    // Edges at 1/4 res (lines are fine at lower res).
    // Synced every frame for smooth panning — drawImage at reduced res is ~0.2ms.
    const container = containerRef.current;
    const sigmaEdges = container.querySelector('canvas.sigma-edges') as HTMLCanvasElement | null;
    const sigmaNodes = container.querySelector('canvas.sigma-nodes') as HTMLCanvasElement | null;

    if (sigmaNodes) {
      // Clean up previous bloom canvases
      nodeBloomRef.current?.remove();
      edgeBloomRef.current?.remove();

      const makeBloom = (className: string, blur: number, opacity: number): [HTMLCanvasElement, CanvasRenderingContext2D] => {
        const c = document.createElement('canvas');
        c.className = className;
        c.style.cssText = `
          position: absolute; inset: 0;
          width: 100%; height: 100%;
          pointer-events: none; z-index: 0;
          filter: blur(${blur}px) brightness(2.0) saturate(1.6);
          opacity: ${opacity};
          mix-blend-mode: screen;
        `;
        container.insertBefore(c, container.firstChild);
        const ctx = c.getContext('2d')!;
        ctx.imageSmoothingEnabled = true;
        return [c, ctx];
      };

      const [nodeBloom, nodeCtx] = makeBloom('sigma-bloom-nodes', 10, nodeGlow);
      const [edgeBloom, edgeCtx] = makeBloom('sigma-bloom-edges', 6, edgeGlow);
      nodeBloomRef.current = nodeBloom;
      edgeBloomRef.current = edgeBloom;

      const NODE_SCALE = 0.5;  // Half res — keeps circles round
      const EDGE_SCALE = 0.5;  // Half res — prevents choppy glow on solid lines

      const syncBloom = () => {
        if (!nodeBloomRef.current || !sigmaNodes) return;

        // Node bloom at 1/2 res
        const nw = Math.ceil(sigmaNodes.width * NODE_SCALE);
        const nh = Math.ceil(sigmaNodes.height * NODE_SCALE);
        if (nodeBloom.width !== nw || nodeBloom.height !== nh) {
          nodeBloom.width = nw;
          nodeBloom.height = nh;
        }
        nodeCtx.clearRect(0, 0, nw, nh);
        nodeCtx.drawImage(sigmaNodes, 0, 0, nw, nh);

        // Edge bloom at 1/4 res
        if (sigmaEdges) {
          const ew = Math.ceil(sigmaEdges.width * EDGE_SCALE);
          const eh = Math.ceil(sigmaEdges.height * EDGE_SCALE);
          if (edgeBloom.width !== ew || edgeBloom.height !== eh) {
            edgeBloom.width = ew;
            edgeBloom.height = eh;
          }
          edgeCtx.clearRect(0, 0, ew, eh);
          edgeCtx.drawImage(sigmaEdges, 0, 0, ew, eh);
        }
      };

      renderer.on('afterRender', syncBloom);
      syncBloom();
    }

    // Bind event handlers directly on this instance
    renderer.on('clickNode', (event: { node: string }) => {
      onNodeClick(event.node);
    });
    renderer.on('doubleClickNode', (event: { node: string }) => {
      onNodeDoubleClick(event.node);
    });
    renderer.on('clickStage', () => {
      onNodeClick('');
    });

    return () => {
      nodeBloomRef.current?.remove();
      edgeBloomRef.current?.remove();
      nodeBloomRef.current = null;
      edgeBloomRef.current = null;
      renderer.kill();
      sigmaRef.current = null;
    };
  }, [graph, nodeCount]); // Re-init when graph object changes OR node count changes

  // Update bloom opacity dynamically without recreating Sigma
  useEffect(() => {
    if (nodeBloomRef.current) nodeBloomRef.current.style.opacity = String(nodeGlow);
    if (edgeBloomRef.current) edgeBloomRef.current.style.opacity = String(edgeGlow);
  }, [nodeGlow, edgeGlow]);

  // Update selected node highlighting
  useEffect(() => {
    sigmaRef.current?.refresh();
  }, [selectedNode, viewState]);

  // Zoom-to-fit highlighted nodes when they change
  const prevHighlightRef = useRef<string>('');
  useEffect(() => {
    const renderer = sigmaRef.current;
    if (!renderer) return;

    const ids = viewState?.highlighted_node_ids ?? [];
    const key = ids.slice().sort().join(',');
    if (key === prevHighlightRef.current || ids.length === 0) {
      // If highlights were cleared, reset camera
      if (ids.length === 0 && prevHighlightRef.current !== '') {
        prevHighlightRef.current = '';
        renderer.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1 }, { duration: 400 });
      }
      return;
    }
    prevHighlightRef.current = key;

    // Collect positions of highlighted nodes in graph coordinates
    const positions: { x: number; y: number }[] = [];
    for (const nodeId of ids) {
      if (graph.hasNode(nodeId)) {
        const attrs = graph.getNodeAttributes(nodeId);
        if (typeof attrs.x === 'number' && typeof attrs.y === 'number') {
          positions.push({ x: attrs.x, y: attrs.y });
        }
      }
    }
    if (positions.length === 0) return;

    // Compute bounding box in graph space
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of positions) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const camera = renderer.getCamera();
    const dims = renderer.getDimensions();

    if (positions.length === 1) {
      // Single node: convert its graph position to viewport, then derive camera center
      const vp = renderer.graphToViewport({ x: cx, y: cy });
      // viewportToFramedGraph gives us the camera-space coords of any viewport point
      const target = renderer.viewportToFramedGraph(vp);
      camera.animate(
        { x: target.x, y: target.y, ratio: Math.min(camera.ratio, 0.3) },
        { duration: 400 },
      );
      return;
    }

    // Multiple nodes: measure how big the bounding box is in viewport pixels at current zoom
    const vpMin = renderer.graphToViewport({ x: minX, y: minY });
    const vpMax = renderer.graphToViewport({ x: maxX, y: maxY });
    const vpSpanX = Math.abs(vpMax.x - vpMin.x);
    const vpSpanY = Math.abs(vpMax.y - vpMin.y);

    // How much we need to scale so the bbox fits in the viewport with padding
    const padding = 1.6;
    const scaleX = vpSpanX > 0 ? (dims.width / (vpSpanX * padding)) : 1;
    const scaleY = vpSpanY > 0 ? (dims.height / (vpSpanY * padding)) : 1;
    const scaleFactor = Math.min(scaleX, scaleY);

    // New ratio: current ratio divided by the needed scale
    const newRatio = Math.max(0.02, Math.min(camera.ratio / scaleFactor, 1.5));

    // Camera target: convert graph center to framed-graph coords
    const vpCenter = renderer.graphToViewport({ x: cx, y: cy });
    const framedCenter = renderer.viewportToFramedGraph(vpCenter);

    camera.animate(
      { x: framedCenter.x, y: framedCenter.y, ratio: newRatio },
      { duration: 500 },
    );
  }, [viewState?.highlighted_node_ids, graph]);

  // Context banner
  const ctx = viewState?.context;
  const hasContext = ctx && (ctx.what || ctx.why || ctx.where);
  const [contextDismissed, setContextDismissed] = React.useState<string | null>(null);

  React.useEffect(() => {
    setContextDismissed(null);
  }, [ctx?.what]);

  // Empty state quote
  const quote = React.useMemo(() => {
    const idx = Math.floor(Math.random() * GOSLING_QUOTES.length);
    return GOSLING_QUOTES[idx];
  }, []);

  return (
    <div className="graph-container">
      {/* Always render at full size — Sigma needs real dimensions for its WebGL canvas */}
      <div
        ref={containerRef}
        className="sigma-container"
        style={{ width: '100%', flex: '1 1 0', minHeight: 0, position: 'relative' }}
      />
      {nodeCount === 0 && (
        <div className="empty-state" style={{ position: 'absolute', inset: 0, zIndex: 5 }}>
          <div className="gosling-quote" style={{ whiteSpace: 'pre-line' }}>
            {quote[0]}
          </div>
          <div className="gosling-sub">{quote[1]}</div>
          <div className="gosling-sub" style={{ marginTop: 20, color: '#2a4a6a' }}>
            ESTABLISHING UPLINK...
          </div>
        </div>
      )}

      {hasContext && contextDismissed !== ctx!.what && (
        <div className="context-banner">
          <button className="cb-dismiss" onClick={() => setContextDismissed(ctx!.what)}>
            &times;
          </button>
          {ctx!.what && (
            <div className="cb-row">
              <span className="cb-label">WHAT</span>
              <span className="cb-text">{ctx!.what}</span>
            </div>
          )}
          {ctx!.why && (
            <div className="cb-row">
              <span className="cb-label">WHY</span>
              <span className="cb-text">{ctx!.why}</span>
            </div>
          )}
          {ctx!.where && (
            <div className="cb-row">
              <span className="cb-label">WHERE</span>
              <span className="cb-text">{ctx!.where}</span>
            </div>
          )}
          {ctx!.source && <div className="cb-source">VIA {ctx!.source.toUpperCase()}</div>}
        </div>
      )}

      <div className="interlinked-watermark">INTERLINKED</div>
    </div>
  );
}
