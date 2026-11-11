// ピンポンカット：ラリーの自動判定（音で見つける）
// 考え方：打球音は「短くて鋭い音」が 1 秒前後の間隔で続く。その並びをラリーとみなす。
//   1) 音を読み取り、低い音（声・足音）を弱めて 5 ミリ秒ごとの音の強さにする
//   2) 鋭く立ち上がってすぐ消える山＝打球音の候補を拾う
//   3) 間隔の近い候補をまとめ、打球数の多いまとまりをラリーにする
// 動画は端末の外に出さない。映像は見ていない（音だけ）。
import { Input, BlobSource, ALL_FORMATS, AudioSampleSink, EncodedPacketSink } from "./vendor/mediabunny-1.61.6.min.js";

export const RATE = 200;      // 音の強さを 1 秒に何個とるか
const CUTOFF_HZ = 1500;       // これより低い音を弱める

// 感度ごとの基準（lo=確実なものだけ / hi=取りこぼしを減らす）
const PRESETS = {
  lo:  { prominence: 14, margin: 14, minHits: 5, maxGap: 1.3 },
  mid: { prominence: 11, margin: 20, minHits: 4, maxGap: 1.5 },
  hi:  { prominence: 8,  margin: 26, minHits: 3, maxGap: 1.8 },
};
const PAD_BEFORE = 0.7, PAD_AFTER = 0.3; // サーブの構え・最後の球の行方ぶん

// 音を少しずつ受け取り、5 ミリ秒ごとの強さをためる
//   mid＝低音を弱めたもの（判定に使う） / all＝そのまま / hi＝5kHz より上（記録用。基準を見直すときの材料）
class Envelope {
  constructor() { this.size = RATE * 600; this.mid = new Float32Array(this.size); this.all = new Float32Array(this.size); this.hi = new Float32Array(this.size); this.length = 0; this.st = [0, 0, 0]; }
  // mono: Float32Array, time: この塊の先頭の時刻（秒）
  push(mono, sampleRate, time) {
    const a1 = 1 / (1 + 2 * Math.PI * CUTOFF_HZ / sampleRate), a2 = 1 / (1 + 2 * Math.PI * 5000 / sampleRate);
    const perFrame = sampleRate / RATE;
    const first = Math.max(0, time) * RATE;
    const lastFrame = Math.floor(first + mono.length / perFrame);
    if (lastFrame >= this.size) {
      this.size = Math.max(this.size * 2, lastFrame + 1);
      for (const k of ["mid", "all", "hi"]) { const bigger = new Float32Array(this.size); bigger.set(this[k]); this[k] = bigger; }
    }
    let [px, y1, y2] = this.st;
    const { mid, all, hi } = this;
    for (let i = 0; i < mono.length; i++) {
      const x = mono[i], f = Math.floor(first + i / perFrame);
      y1 = a1 * (y1 + x - px); y2 = a2 * (y2 + x - px); px = x;
      mid[f] += y1 * y1; all[f] += x * x; hi[f] += y2 * y2;
    }
    this.st = [px, y1, y2];
    if (lastFrame + 1 > this.length) this.length = lastFrame + 1;
  }
  result() { const n = this.length; return { mid: this.mid.subarray(0, n), all: this.all.subarray(0, n), hi: this.hi.subarray(0, n) }; }
}

// AAC の音を約 20 秒ずつの小さなファイル（ADTS 形式）にして順に渡す。
// WebCodecs で音を読めないブラウザ（Safari など）でも、ブラウザ標準の機能で少しずつ読めるようにするため。
// 継ぎ目の乱れを避けるため、各かたまりの頭に前のかたまりの終わりを少し重ねる（newStart より前は捨てる）
const CHUNK_SEC = 20, OVERLAP = 4;
export async function* aacChunks(track) {
  const cfg = await track.getDecoderConfig();
  const d = cfg && cfg.description;
  if (!d) throw new Error("音の形式の情報がありません");
  const asc = d instanceof ArrayBuffer ? new Uint8Array(d) : new Uint8Array(d.buffer, d.byteOffset, d.byteLength);
  let type = asc[0] >> 3; const freq = ((asc[0] & 7) << 1) | (asc[1] >> 7), chan = (asc[1] >> 3) & 15;
  if (type === 5 || type === 29) type = 2; // HE-AAC は土台の部分だけ読む
  if (type < 1 || type > 4 || freq > 12 || !chan) throw new Error("対応していない AAC です");
  const wrap = list => {
    const out = new Uint8Array(list.reduce((n, p) => n + p.data.byteLength + 7, 0));
    let o = 0;
    for (const p of list) {
      const L = p.data.byteLength + 7;
      out.set([0xFF, 0xF1, ((type - 1) << 6) | (freq << 2) | (chan >> 2), ((chan & 3) << 6) | (L >> 11), (L >> 3) & 0xFF, ((L & 7) << 5) | 0x1F, 0xFC], o);
      out.set(p.data, o + 7); o += L;
    }
    return out;
  };
  const sink = new EncodedPacketSink(track);
  let list = [], fresh = 0; // fresh＝list のうち新しく入った最初の位置
  for await (const p of sink.packets()) {
    list.push(p);
    if (p.timestamp - list[fresh].timestamp >= CHUNK_SEC) {
      yield { bytes: wrap(list), start: list[0].timestamp, newStart: list[fresh].timestamp };
      list = list.slice(-OVERLAP); fresh = list.length;
    }
  }
  if (list.length > fresh) yield { bytes: wrap(list), start: list[0].timestamp, newStart: list[fresh].timestamp };
}
// 読み取った 1 かたまりを、重ねた分を捨ててためる
export function pushChunk(env, chunk, mono, sampleRate) {
  const skip = Math.min(mono.length, Math.round((chunk.newStart - chunk.start) * sampleRate));
  env.push(mono.subarray(skip), sampleRate, chunk.newStart);
}
export const newEnvelope = () => new Envelope();

function decodeWith(ctx, arrayBuffer) {
  return new Promise((ok, ng) => { try { const p = ctx.decodeAudioData(arrayBuffer, ok, ng); if (p && p.catch) p.catch(ng); } catch (e) { ng(e); } });
}
function toMono(buf) {
  const mono = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) mono[i] += d[i]; }
  return mono;
}

/**
 * 動画の音を読み取り、音の強さの列を返す（感度を変えて判定し直すときに使い回せる）
 * @param {File} file
 * @param {(ratio:number)=>void} onProgress
 * @param {()=>boolean} isCancelled true を返すと中断（null を返す）
 * @returns {Promise<{mid:Float32Array,all:Float32Array,hi:Float32Array}|null>}
 */
export async function analyzeAudio(file, onProgress = () => {}, isCancelled = () => false) {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const errs = [];
  let lastYield = performance.now();
  const breathe = async ratio => { // 画面が固まらないよう時々ゆずる。中断されていたら true
    if (performance.now() - lastYield > 100) { onProgress(Math.min(1, ratio)); await new Promise(r => setTimeout(r)); lastYield = performance.now(); }
    return isCancelled();
  };
  try {
    const track = await input.getPrimaryAudioTrack();
    if (!track) throw Object.assign(new Error("この動画には音が入っていないため、自動判定できません"), { noAudio: true });
    const duration = (await track.computeDuration()) || 1;

    // ① WebCodecs で少しずつ読む（Chrome など）
    if (typeof AudioDecoder !== "undefined" && (await track.canDecode())) {
      try {
        const env = new Envelope();
        const sink = new AudioSampleSink(track);
        let plane = new Float32Array(0), mono = new Float32Array(0);
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
          if (await breathe(t / duration)) return null;
        }
        return env.result();
      } catch (e) { errs.push("①" + e.message); }
    } else errs.push("①未対応");

    // ② AAC を小さなかたまりに分け、ブラウザ標準の機能で読む（Safari など）
    if (track.codec === "aac" && Ctx) {
      try {
        const env = new Envelope();
        const ctx = new Ctx(1, 1, track.sampleRate || 44100);
        for await (const chunk of aacChunks(track)) {
          const buf = await decodeWith(ctx, chunk.bytes.buffer);
          pushChunk(env, chunk, toMono(buf), buf.sampleRate);
          if (await breathe(chunk.newStart / duration)) return null;
        }
        if (env.length) return env.result();
        errs.push("②音が空");
      } catch (e) { errs.push("②" + ((e && e.message) || e)); }
    }
  } catch (err) {
    if (err && err.noAudio) throw err;
    errs.push((err && err.message) || String(err));
  } finally {
    try { input.dispose && input.dispose(); } catch (e) {}
  }
  // ③ 控え：丸ごと読み取る（メモリを使うので大きい動画は不可）
  const fail = () => new Error("この端末では音を読み取れませんでした（" + errs.join(" / ") + "）");
  if (!Ctx || file.size > 600 * 1048576) throw fail();
  let buf;
  try { buf = await decodeWith(new Ctx(1, 1, 22050), await file.arrayBuffer()); } catch (e) { errs.push("③" + ((e && e.message) || e)); throw fail(); }
  if (isCancelled()) return null;
  const env = new Envelope();
  env.push(toMono(buf), buf.sampleRate, 0);
  return env.result();
}

// うまく判定できなかったときに送ってもらう記録（音そのものではなく、5 ミリ秒ごとの音の大きさだけ）
export function packRecord(env, extra) {
  const pack = arr => {
    const q = new Uint8Array(arr.length); // 0.5dB きざみ、-100dB〜+27dB
    for (let i = 0; i < arr.length; i++) q[i] = Math.max(0, Math.min(255, Math.round((10 * Math.log10(arr[i] + 1e-12) + 100) * 2)));
    let bin = ""; for (let i = 0; i < q.length; i += 8192) bin += String.fromCharCode.apply(null, q.subarray(i, i + 8192));
    return btoa(bin);
  };
  return JSON.stringify({ version: 1, rate: RATE, ...extra, mid: pack(env.mid), all: pack(env.all), hi: pack(env.hi) });
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
