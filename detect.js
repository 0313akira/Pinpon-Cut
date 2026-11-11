// ピンポンカット：ラリーの自動判定（音で見つける）
// 考え方：打球音は「短くて鋭い音」が 1 秒前後の間隔で続く。その並びをラリーとみなす。
//   1) 音を読み取り、低い音（声・足音）を弱めて 5 ミリ秒ごとの音の強さにする
//   2) 鋭く立ち上がってすぐ消える山＝打球音の候補を拾う
//   3) 間隔の近い候補をまとめ、打球数の多いまとまりをラリーにする
// 動画は端末の外に出さない。映像は見ていない（音だけ）。
import { Input, BlobSource, ALL_FORMATS, AudioSampleSink } from "./vendor/mediabunny-1.61.6.min.js";

export const RATE = 200;      // 音の強さを 1 秒に何個とるか
const CUTOFF_HZ = 1500;       // これより低い音を弱める

// 感度ごとの基準（lo=確実なものだけ / hi=取りこぼしを減らす）
const PRESETS = {
  lo:  { prominence: 14, margin: 14, minHits: 5, maxGap: 1.3 },
  mid: { prominence: 11, margin: 20, minHits: 4, maxGap: 1.5 },
  hi:  { prominence: 8,  margin: 26, minHits: 3, maxGap: 1.8 },
};
const PAD_BEFORE = 0.7, PAD_AFTER = 0.3; // サーブの構え・最後の球の行方ぶん

// 音を少しずつ受け取り、5 ミリ秒ごとの強さ（低音を弱めたあとのエネルギー）をためる
class Envelope {
  constructor() { this.data = new Float32Array(RATE * 600); this.length = 0; this.px = 0; this.py = 0; }
  // mono: Float32Array, time: この塊の先頭の時刻（秒）
  push(mono, sampleRate, time) {
    const alpha = 1 / (1 + 2 * Math.PI * CUTOFF_HZ / sampleRate);
    const perFrame = sampleRate / RATE;
    const first = Math.max(0, time) * RATE;
    const lastFrame = Math.floor(first + mono.length / perFrame);
    if (lastFrame >= this.data.length) {
      const bigger = new Float32Array(Math.max(this.data.length * 2, lastFrame + 1));
      bigger.set(this.data); this.data = bigger;
    }
    let px = this.px, py = this.py;
    const data = this.data;
    for (let i = 0; i < mono.length; i++) {
      const x = mono[i];
      py = alpha * (py + x - px); px = x;
      data[Math.floor(first + i / perFrame)] += py * py;
    }
    this.px = px; this.py = py;
    if (lastFrame + 1 > this.length) this.length = lastFrame + 1;
  }
  result() { return this.data.subarray(0, this.length); }
}

/**
 * 動画の音を読み取り、音の強さの列を返す（感度を変えて判定し直すときに使い回せる）
 * @param {File} file
 * @param {(ratio:number)=>void} onProgress
 * @param {()=>boolean} isCancelled true を返すと中断（null を返す）
 * @returns {Promise<Float32Array|null>}
 */
export async function analyzeAudio(file, onProgress = () => {}, isCancelled = () => false) {
  const env = new Envelope();
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  let streamErr = null;
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) throw Object.assign(new Error("この動画には音が入っていないため、自動判定できません"), { noAudio: true });
    if (typeof AudioDecoder === "undefined" || !(await track.canDecode())) throw new Error("音（" + track.codec + "）を読み取れません");
    const duration = (await track.computeDuration()) || 1;
    const sink = new AudioSampleSink(track);
    let plane = new Float32Array(0), mono = new Float32Array(0), lastYield = performance.now();
    for await (const s of sink.samples()) {
      const n = s.numberOfFrames, ch = s.numberOfChannels;
      if (plane.length < n) { plane = new Float32Array(n); mono = new Float32Array(n); }
      const m = mono.subarray(0, n), p = plane.subarray(0, n);
      s.copyTo(m, { planeIndex: 0, format: "f32-planar" });
      for (let c = 1; c < ch; c++) {
        s.copyTo(p, { planeIndex: c, format: "f32-planar" });
        for (let i = 0; i < n; i++) m[i] += p[i];
      }
      env.push(m, s.sampleRate, s.timestamp);
      const t = s.timestamp; s.close();
      if (performance.now() - lastYield > 100) { // 画面が固まらないよう時々ゆずる
        onProgress(Math.min(1, t / duration));
        await new Promise(r => setTimeout(r));
        lastYield = performance.now();
        if (isCancelled()) return null;
      }
    }
    return env.result();
  } catch (err) {
    if (err && err.noAudio) throw err;
    streamErr = err;
  } finally {
    try { input.dispose && input.dispose(); } catch (e) {}
  }
  // 控え：ブラウザ標準の機能で丸ごと読み取る（メモリを使うので大きい動画は不可）
  if (file.size > 600 * 1048576) throw new Error("この端末では音を読み取れませんでした（" + streamErr.message + "）");
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new Ctx(1, 1, 22050);
  const buf = await new Promise(async (ok, ng) => {
    try { const p = ctx.decodeAudioData(await file.arrayBuffer(), ok, ng); if (p && p.catch) p.catch(ng); } catch (e) { ng(e); }
  }).catch(() => { throw new Error("この端末では音を読み取れませんでした（" + streamErr.message + "）"); });
  if (isCancelled()) return null;
  const mono = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) mono[i] += d[i]; }
  const fallback = new Envelope();
  fallback.push(mono, buf.sampleRate, 0);
  return fallback.result();
}

// 打球音の候補（時刻と目立ち具合）を拾う
function findHits(env, prominenceMin) {
  const n = env.length, db = new Float32Array(n);
  for (let i = 0; i < n; i++) db[i] = 10 * Math.log10(env[i] + 1e-12);

  // まわり（前後 1.5 秒）のふだんの音の大きさ。0.25 秒おきに中央値をとる
  const step = RATE / 4, half = RATE * 1.5, floors = [];
  for (let c = 0; c < n; c += step) {
    const w = Array.from(db.subarray(Math.max(0, c - half), Math.min(n, c + half))).sort((a, b) => a - b);
    floors.push(w[w.length >> 1]);
  }
  const floorAt = i => floors[Math.min(floors.length - 1, Math.round(i / step))];

  const hits = [];
  const NEAR = 16; // 80 ミリ秒以内でいちばん大きい所だけ
  for (let i = 8; i < n - 20; i++) {
    const v = db[i];
    let isPeak = true;
    for (let k = 1; k <= NEAR && isPeak; k++) if (db[i - k] > v || (i + k < n && db[i + k] >= v)) isPeak = false;
    if (!isPeak) continue;
    const prominence = v - floorAt(i);
    if (prominence < prominenceMin) continue;
    // 鋭い立ち上がり（10〜40 ミリ秒前は静か）
    let before = Infinity; for (let k = 2; k <= 8; k++) before = Math.min(before, db[i - k]);
    if (v - before < 8) continue;
    // すぐ消える（60〜100 ミリ秒後には小さい）。声や歓声は長く続くので落ちる
    let after = 0; for (let k = 12; k <= 20; k++) after += db[i + k];
    if (v - after / 9 < 5) continue;
    hits.push({ t: i / RATE, prominence });
  }
  return hits;
}

/**
 * 音の強さの列からラリーの区間を出す
 * @param {Float32Array} env analyzeAudio の結果
 * @param {"lo"|"mid"|"hi"} sensitivity
 * @param {number} duration 動画の長さ（秒）
 * @returns {{start:number,end:number,hits:number}[]}
 */
export function findRallies(env, sensitivity = "mid", duration = Infinity) {
  const p = PRESETS[sensitivity] || PRESETS.mid;
  let hits = findHits(env, p.prominence);
  if (!hits.length) return [];
  // 自分の台の打球音はいちばん大きい部類。それよりずっと小さい音（となりの台など）は外す
  const sorted = hits.map(h => h.prominence).sort((a, b) => a - b);
  const loud = sorted[Math.floor(sorted.length * 0.9)];
  hits = hits.filter(h => h.prominence >= loud - p.margin);

  const rallies = [];
  let group = [hits[0]];
  const flush = () => {
    if (group.length >= p.minHits) {
      const start = Math.max(0, group[0].t - PAD_BEFORE), end = Math.min(duration, group[group.length - 1].t + PAD_AFTER);
      const last = rallies[rallies.length - 1];
      if (last && start <= last.end) { last.end = end; last.hits += group.length; }
      else rallies.push({ start, end, hits: group.length });
    }
  };
  for (let i = 1; i < hits.length; i++) {
    if (hits[i].t - hits[i - 1].t <= p.maxGap) group.push(hits[i]);
    else { flush(); group = [hits[i]]; }
  }
  flush();
  return rallies;
}
