// Snake: the board is a Hamiltonian cycle, so following it forever is always safe. Each cell the
// code works out which of the four directions are legal, which stay safe (never cross the tail on
// the cycle, never skip the food on the safe route) and which reaches the food, writes the four
// verdicts into one sentence each, and asks Laya to pick a direction. A deterministic shield keeps
// a truly unsafe pick off the board; everything else, good or bad, is played. Ported from the
// laya-mlx terminal demo's game.py + policy.py (compact prompt), one question first so it records.
import { rng } from '../lib3d.js';

const W = 24, H = 16, CAP = W * H, LEN0 = 6, MOVE = 0.1;   // board, start length, seconds per cell step
const DIRS = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
const VEC = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
const OPP = { UP: 'DOWN', DOWN: 'UP', LEFT: 'RIGHT', RIGHT: 'LEFT' };
const key = (x, y) => y * W + x;   // a cell as one integer
const range = (a, b) => { const r = [], s = a < b ? 1 : -1; for (let i = a; i !== b; i += s) r.push(i); return r; };

function cycle(w, h) {   // visit every square once with adjacent steps, closing edge included
  if (Math.min(w, h) < 4 || (w % 2 && h % 2)) throw new Error('board needs >= 4 with one even side');
  if (h % 2) return cycle(h, w).map(([x, y]) => [y, x]);
  const path = [[0, 0]];
  for (let y = 0; y < h; y++) for (const x of (y % 2 === 0 ? range(1, w) : range(w - 1, 0))) path.push([x, y]);
  for (let y = h - 1; y > 0; y--) path.push([0, y]);
  return path;
}
const CYCLE = cycle(W, H);
const IDX = new Map(CYCLE.map(([x, y], i) => [key(x, y), i]));
const idx = (x, y) => IDX.get(key(x, y));
const dirFrom = (a, b) => DIRS.find(d => a[0] + VEC[d][0] === b[0] && a[1] + VEC[d][1] === b[1]);

class Snake {
  constructor(seed) { this.seed = seed >>> 0; this.best = 0; this.crashes = 0; this.shields = 0; this.reset(); }

  reset() {
    this.rand = rng((this.seed + this.crashes * 0x9e3779b1) >>> 0);   // a fresh food stream each life, still deterministic
    this.t = 0; this.acc = 0; this.score = 0; this.dead = 0; this.won = false; this.shield = false;
    const start = idx(W >> 1, H >> 1);
    this.body = Array.from({ length: LEN0 }, (_, i) => CYCLE[(start - i + CAP) % CAP]);   // [head, ...tail]
    this.set = new Set(this.body.map(([x, y]) => key(x, y)));
    this.dir = this.moved = dirFrom(this.body[0], CYCLE[(start + 1) % CAP]);   // moved = last executed step, for the YOU PLAY reverse guard
    this.food = this.spawnFood();
  }

  get head() { return this.body[0]; }
  spawnFood() { const empty = CYCLE.filter(([x, y]) => !this.set.has(key(x, y))); return empty.length ? empty[this.rand() * empty.length | 0] : null; }

  legalReason(dir) {   // wall / reverse / body / legal, with the tail freed on a non-growing step
    const [x, y] = [this.head[0] + VEC[dir][0], this.head[1] + VEC[dir][1]], k = key(x, y);
    if (x < 0 || x >= W || y < 0 || y >= H) return 'wall';
    if (this.body[1] && k === key(...this.body[1])) return 'reverse';
    const occupied = new Set(this.set);
    if (this.food && k !== key(...this.food)) occupied.delete(key(...this.body[this.body.length - 1]));
    return occupied.has(k) ? 'body' : 'legal';
  }

  moves() {   // per direction: legal, safe (cycle-safe), how far it advances on the cycle, whether it eats
    if (this.dead || this.won || !this.food) return [];
    const hi = idx(...this.head), tailDist = (idx(...this.body[this.body.length - 1]) - hi + CAP) % CAP, foodDist = (idx(...this.food) - hi + CAP) % CAP;
    return DIRS.map(dir => {
      const reason0 = this.legalReason(dir), legal = reason0 === 'legal';
      const [tx, ty] = [this.head[0] + VEC[dir][0], this.head[1] + VEC[dir][1]], on = tx >= 0 && tx < W && ty >= 0 && ty < H;   // key() only injective on the board, so bounds-check before the cycle lookup
      const ti = on ? idx(tx, ty) : hi, advance = (ti - hi + CAP) % CAP, eats = on && this.food && key(tx, ty) === key(...this.food);
      let safe = legal, reason = reason0;
      if (safe && (advance > tailDist || (advance === tailDist && eats))) { safe = false; reason = 'would cross the tail'; }
      if (safe && (advance === 0 || advance > foodDist)) { safe = false; reason = 'would skip the food on the safe route'; }
      return { dir, legal, safe, advance, eats, reason };
    });
  }

  foodReachability() {   // empty-cell connectivity from the head; the tail is not treated as empty
    const blocked = new Set(this.set); blocked.delete(key(...this.head));
    const seen = new Set([key(...this.head)]), q = [this.head];
    for (let i = 0; i < q.length; i++) { const [x, y] = q[i];
      for (const d of DIRS) { const nx = x + VEC[d][0], ny = y + VEC[d][1], k = key(nx, ny);
        if (nx >= 0 && nx < W && ny >= 0 && ny < H && !blocked.has(k) && !seen.has(k)) { seen.add(k); q.push([nx, ny]); } } }
    return { reachable: !!this.food && seen.has(key(...this.food)), space: seen.size };
  }

  preferred(moves) { const safe = moves.filter(m => m.safe); return safe.length ? safe.reduce((a, b) => b.advance > a.advance ? b : a).dir : 'NONE'; }

  step(dir) {
    if (this.legalReason(dir) !== 'legal') { if (!this.dead) { this.dead = 1.1; this.crashes++; } return; }
    const target = [this.head[0] + VEC[dir][0], this.head[1] + VEC[dir][1]], eats = this.food && key(...target) === key(...this.food);
    if (!eats) { const tail = this.body.pop(); this.set.delete(key(...tail)); }   // tail first: the head may land on the cell the tail vacates
    this.body.unshift(target); this.set.add(key(...target)); this.moved = dir;
    if (eats) {
      this.score++; this.best = Math.max(this.best, this.score);
      if (this.body.length === CAP) { this.won = true; this.dead = 1.4; this.food = null; } else this.food = this.spawnFood();
    }
  }

  update(dt, input) {
    this.t += dt;
    if (this.dead) { this.dead -= dt; if (this.dead <= 0) this.reset(); return; }
    if (input && input.dir && input.dir !== OPP[this.moved]) this.dir = input.dir;   // YOU PLAY: steer, never back into the neck (guard the last executed step, not a pending turn)
    for (this.acc += dt; this.acc >= MOVE && !this.dead; this.acc -= MOVE) this.step(this.dir);
  }

  observe() {   // the compact prompt from policy.py: one sentence of state, the move choice first so it records, then two nouls
    const moves = this.moves(), safe = moves.filter(m => m.safe), { reachable } = this.foodReachability(), best = this.preferred(moves), criteria = {};
    for (const m of moves) criteria[m.dir] = !m.legal ? 'Blocked. Collision.' : !m.safe ? 'Unsafe. Traps the snake.'
      : m.eats ? 'Safe. Eat food now. Best.' : m.dir === best ? 'Safe. Best route to food.' : 'Safe. Slower route.';
    return {
      state: `Safe route: ${safe.length ? 'yes' : 'no'}. Food reachable through empty cells: ${reachable ? 'yes' : 'no'}.`,
      questions: {
        move: { type: 'choice', instructions: 'Choose the best safe move toward food.', criteria },
        risk: { type: 'noul', instructions: 'Is a safe route available?' },
        food: { type: 'noul', instructions: 'Is food reachable through empty cells?' },
      },
    };
  }

  act(answers, params, apply) {   // the model ranks the four directions; the shield keeps an unsafe pick off the board
    const p = answers.move && answers.move.probabilities;
    if (this.dead || !p) return { label: '—', why: this.dead ? 'crashed, waiting to respawn' : 'no answer yet' };
    const by = d => p[d] ?? -1, proposed = DIRS.reduce((a, b) => by(b) > by(a) ? b : a);
    const safe = this.moves().filter(m => m.safe).map(m => m.dir);
    const executed = safe.includes(proposed) ? proposed : safe.length ? safe.reduce((a, b) => by(b) > by(a) ? b : a) : proposed;
    const intervened = executed !== proposed;
    if (apply) { this.dir = executed; if (intervened) this.shields++; }
    return intervened
      ? { label: executed, why: `${proposed} was unsafe, the shield took ${executed} · ${this.shields} shielded so far` }
      : { label: executed, why: safe.length ? `P(${executed}) ${by(executed).toFixed(2)}, the safe way toward food` : 'no safe move left' };
  }

  draw(ctx, w, h) {   // top-down, grayscale; the stage dithers it to ink and amber
    const cell = Math.min(w / W, h / H), ox = (w - cell * W) / 2, oy = (h - cell * H) / 2, pad = Math.max(1, cell * 0.09);
    ctx.fillStyle = 'rgb(9,9,9)'; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgb(16,16,16)'; ctx.fillRect(ox, oy, cell * W, cell * H);
    ctx.fillStyle = 'rgb(26,26,26)';
    for (let x = 0; x <= W; x++) ctx.fillRect(ox + x * cell - 0.5, oy, 1, cell * H);
    for (let y = 0; y <= H; y++) ctx.fillRect(ox, oy + y * cell - 0.5, cell * W, 1);
    const rect = (cx, cy, g, inset = pad) => { ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(ox + cx * cell + inset, oy + cy * cell + inset, cell - 2 * inset, cell - 2 * inset); };
    if (this.food) rect(this.food[0], this.food[1], Math.min(255, 205 + Math.round(45 * (0.5 + 0.5 * Math.sin(this.t * 6)))), pad * 1.5);
    if (!(this.dead && !this.won && Math.floor(this.dead * 12) % 2)) {   // blink while crashed, but a win stays lit
      const n = this.body.length;
      this.body.forEach(([x, y], i) => rect(x, y, i === 0 ? 255 : Math.round(150 + 95 * (1 - i / n))));
    }
  }
}

export default {
  id: 'snake', title: 'Snake', checkpoint: 'multilingual', keys: '← ↑ → ↓ TO STEER',
  blurb: 'The board is one long safe loop. Each cell the code labels the four directions and Laya picks one; a shield keeps a fatal pick off the board.',
  params: [],
  input(keys) {
    const dir = keys.has('ArrowUp') ? 'UP' : keys.has('ArrowDown') ? 'DOWN' : keys.has('ArrowLeft') ? 'LEFT' : keys.has('ArrowRight') ? 'RIGHT' : null;
    return { dir };
  },
  create: seed => new Snake(seed),
};
