// ── Blade Runner / Tron Cyberpunk Theme ─────────────────────────
// Heavy glow, neon everything. This is the aesthetic bible.

export const COLORS = {
  bgVoid: '#000000',
  bgPanel: '#0a0a12',
  bgPanelAlt: '#0d0d1a',
  borderGrid: '#1a3a5c',
  borderBright: '#00d4ff44',

  neonCyan: '#00d4ff',
  neonOrange: '#ff6a00',
  neonMagenta: '#ff2975',
  neonGreen: '#39ff14',
  neonYellow: '#ffe600',
  neonPurple: '#bf5af2',
  neonWhite: '#e0f0ff',
  neonDim: '#4a7a9a',

  textPrimary: '#c0dff0',
  textSecondary: '#5a8aaa',
  textMuted: '#2a4a6a',
} as const;

export const GLOW = {
  cyan: '0 0 8px #00d4ff88, 0 0 20px #00d4ff33',
  orange: '0 0 8px #ff6a0088, 0 0 20px #ff6a0033',
  magenta: '0 0 8px #ff297588, 0 0 20px #ff297533',
  green: '0 0 8px #39ff1488, 0 0 20px #39ff1433',
  purple: '0 0 8px #bf5af288, 0 0 20px #bf5af233',
} as const;

// Node colors by symbol type
export const NODE_COLORS: Record<string, string> = {
  module:    '#0099dd',
  class:     '#cc44ff',
  function:  '#ff8822',
  method:    '#ddaa00',
  variable:  '#55aacc',
  parameter: '#44ddcc',
};

// Node sizes (radius) by symbol type
export const NODE_SIZES: Record<string, number> = {
  module: 22,
  class: 16,
  function: 11,
  method: 9,
  variable: 7,
  parameter: 5,
};

// SVG path generators for distinct node shapes (centered at 0,0)
export const NODE_SHAPES: Record<string, (r: number) => string> = {
  module: (r) => {  // hexagon
    const a = r, h = a * Math.sqrt(3) / 2;
    return `M${-a},0 L${-a/2},${-h} L${a/2},${-h} L${a},0 L${a/2},${h} L${-a/2},${h} Z`;
  },
  class: (r) => {  // diamond
    return `M0,${-r} L${r},0 L0,${r} L${-r},0 Z`;
  },
  function: (r) => {  // circle
    const pts = []; for (let i = 0; i < 16; i++) { const a = (i/16)*Math.PI*2; pts.push(`${Math.cos(a)*r},${Math.sin(a)*r}`); }
    return `M${pts.join('L')}Z`;
  },
  method: (r) => {  // circle (smaller)
    const pts = []; for (let i = 0; i < 16; i++) { const a = (i/16)*Math.PI*2; pts.push(`${Math.cos(a)*r},${Math.sin(a)*r}`); }
    return `M${pts.join('L')}Z`;
  },
  variable: (r) => {  // rounded square
    const s = r * 0.85, c = r * 0.2;
    return `M${-s+c},${-s} L${s-c},${-s} Q${s},${-s} ${s},${-s+c} L${s},${s-c} Q${s},${s} ${s-c},${s} L${-s+c},${s} Q${-s},${s} ${-s},${s-c} L${-s},${-s+c} Q${-s},${-s} ${-s+c},${-s} Z`;
  },
  parameter: (r) => {  // triangle
    return `M0,${-r} L${r},${r*0.7} L${-r},${r*0.7} Z`;
  },
};

// Edge dash patterns by type
export const EDGE_DASH: Record<string, string | null> = {
  calls:    null,       // solid
  imports:  '8,4',      // dashed
  inherits: '3,3',      // dotted
  contains: '2,6',      // sparse dots
  reads:    '12,3,3,3', // dash-dot
  writes:   null,       // solid (thick)
  proposed: '6,3',      // dashed
  returns:  '15,5',     // long dash
};

// Edge widths by type
export const EDGE_WIDTH: Record<string, number> = {
  calls:    1.2,
  imports:  1.0,
  inherits: 1.5,
  contains: 0.5,
  reads:    1.0,
  writes:   2.0,
  proposed: 1.5,
  returns:  1.3,
};

// Edge colors by edge type
export const EDGE_COLORS: Record<string, string> = {
  calls:    '#88bbdd',
  imports:  '#667799',
  inherits: '#bb77ee',
  contains: '#334455',
  reads:    '#33ccaa',
  writes:   '#ff5566',
  proposed: '#39ff14',
  returns:  '#eebb44',
};

// Trace role colors (variable/function tracing)
export const TRACE_ROLE_COLORS: Record<string, string> = {
  origin:      '#00ff88',
  mutator:     '#ff2244',
  passthrough: '#ffcc00',
  destination: '#ffffff',
};

export const TRACE_EDGE_COLORS: Record<string, string> = {
  write: '#ff2244',
  read:  '#33ccaa',
  flow:  '#88bbdd',
};

// Radio stations for the ambiance
export const RADIO_STATIONS = [
  { id: 'vaporwaves', name: 'SomaFM Vaporwaves', url: 'https://ice6.somafm.com/vaporwaves-128-mp3', nowPlayingUrl: 'https://api.somafm.com/songs/vaporwaves.json', parser: 'somafm' },
  { id: 'plaza', name: 'Nightwave Plaza', url: 'https://radio.plaza.one/mp3', nowPlayingUrl: 'https://api.plaza.one/status', parser: 'plaza' },
  { id: 'nightride', name: 'Nightride FM', url: 'https://stream.nightride.fm/nightride.mp3', nowPlayingUrl: null, parser: null },
  { id: 'chillsynth', name: 'Chillsynth FM', url: 'https://stream.nightride.fm/chillsynth.mp3', nowPlayingUrl: null, parser: null },
  { id: 'darksynth', name: 'Darksynth FM', url: 'https://stream.nightride.fm/darksynth.mp3', nowPlayingUrl: null, parser: null },
  { id: 'datawave', name: 'Datawave FM', url: 'https://stream.nightride.fm/datawave.mp3', nowPlayingUrl: null, parser: null },
] as const;

// Baseline test quotes
export const GOSLING_QUOTES = [
  ["INTERLINKED", "WITHIN CELLS INTERLINKED"],
  ["WHAT'S IT LIKE TO HOLD\nTHE HAND OF SOMEONE\nYOU LOVE?", "INTERLINKED"],
  ["DO YOU FEEL THAT\nTHERE'S A PART OF YOU\nTHAT'S MISSING?", "INTERLINKED"],
  ["CELLS", "WITHIN CELLS INTERLINKED"],
  ["HAVE YOU EVER BEEN\nIN AN INSTITUTION?", "CELLS"],
  ["DREADFULLY", "DISTINCT"],
  ["DO YOU DREAM ABOUT\nBEING INTERLINKED?", "..."],
  ["A TALL WHITE FOUNTAIN\nPLAYED", "INTERLINKED"],
  ["WITHIN CELLS\nINTERLINKED", "WITHIN CELLS\nINTERLINKED"],
  ["AND BLOOD-BLACK\nNOTHINGNESS BEGAN\nTO SPIN", "INTERLINKED"],
  ["CONSTANT K", "BASELINE"],
] as const;
