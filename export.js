// ピンポンカット：高速書き出し（再生せずにコマを組み替える）
// 映像：必要なコマだけ読み取り → 作り直し → MP4 に組み立て
// 音声：作り直さず、元の音のかけらをそのまま並べる（音ズレしない）
// 部品：mediabunny（MPL-2.0, vendor/ に同梱。外部へは何も送らない）
import {
  Input, BlobSource, ALL_FORMATS, Output, Mp4OutputFormat, BufferTarget,
  VideoSampleSink, VideoSampleSource, EncodedPacketSink, EncodedAudioPacketSource,
  EncodedPacket, getFirstEncodableVideoCodec, QUALITY_HIGH,
} from "./vendor/mediabunny-1.61.6.min.js";

// 書き出せる環境か（WebCodecs が無い古いブラウザは false）
export function fastCutSupported() {
  return typeof VideoEncoder !== "undefined" && typeof VideoDecoder !== "undefined";
}

/**
 * @param {File} file 元の動画
 * @param {{start:number,end:number}[]} segs 残す区間（時刻順・重なりなし）
 * @param {{maxSide?:number}} opt maxSide=長い辺の上限（0 で元のまま）
 * @param {(ratio:number)=>void} onProgress
 * @returns {Promise<Blob>} MP4
 */
export async function fastCut(file, segs, opt = {}, onProgress = () => {}) {
  const diag = { step: "開く", seg: 0 };
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  try {
    const vt = await input.getPrimaryVideoTrack();
    if (!vt) throw new Error("映像が見つかりません");
    const at = await input.getPrimaryAudioTrack();
    diag.v = vt.codec; diag.a = at ? at.codec : "なし";
    if (!(await vt.canDecode())) throw new Error("この端末では元の映像（" + vt.codec + "）を読み取れません");

    // 出力サイズ（偶数にそろえる）
    const dw = await vt.getDisplayWidth(), dh = await vt.getDisplayHeight();
    const scale = opt.maxSide ? Math.min(1, opt.maxSide / Math.max(dw, dh)) : 1;
    const outW = Math.round(dw * scale / 2) * 2, outH = Math.round(dh * scale / 2) * 2;

    // フレームレート：映像の時刻を「等間隔の格子」に載せるために使う
    const stats = await vt.computePacketStats(120);
    let fps = stats.averagePacketRate || 30;
    fps = [24, 25, 30, 48, 50, 60, 120].reduce((a, b) => Math.abs(b - fps) < Math.abs(a - fps) ? b : a);
    diag.fps = fps; diag.size = outW + "x" + outH;

    diag.step = "書き出し方式を選ぶ";
    const codec = await getFirstEncodableVideoCodec(["avc", "hevc", "vp9", "av1"], { width: outW, height: outH });
    if (!codec) throw new Error("この端末では映像を作り直せません");
    diag.out = codec;

    const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target: new BufferTarget() });
    const vsrc = new VideoSampleSource({
      codec, bitrate: QUALITY_HIGH, keyFrameInterval: 2,
      transform: { width: outW, height: outH, fit: "fill", frameRate: fps },
    });
    output.addVideoTrack(vsrc, { frameRate: fps });

    let asrc = null, aSink = null, aCfg = null;
    if (at) {
      aCfg = await at.getDecoderConfig();
      asrc = new EncodedAudioPacketSource(at.codec);
      output.addAudioTrack(asrc);
      aSink = new EncodedPacketSink(at);
    }
    await output.start();

    const total = segs.reduce((s, x) => s + x.end - x.start, 0) || 1;
    let offset = 0, firstAudio = true;

    for (let i = 0; i < segs.length; i++) {
      diag.seg = i + 1; diag.step = "区間の準備";
      let a = segs[i].start, b = segs[i].end;

      // 区間の境目を音のかけら（約20ミリ秒）にそろえる
      let pkts = [];
      if (aSink) {
        let p = (await aSink.getPacket(a)) || (await aSink.getFirstPacket());
        while (p && p.timestamp < b) { pkts.push(p); p = await aSink.getNextPacket(p); }
        if (pkts.length) {
          a = pkts[0].timestamp;
          const last = pkts[pkts.length - 1];
          b = last.timestamp + last.duration;
        }
      }
      const len = b - a;
      let ai = 0;
      const pushAudioUntil = async (tOut) => {
        while (ai < pkts.length && offset + pkts[ai].timestamp - a <= tOut) {
          const p = pkts[ai++];
          const np = new EncodedPacket(p.data, p.type, offset + p.timestamp - a, p.duration);
          await asrc.add(np, firstAudio ? { decoderConfig: aCfg } : undefined);
          firstAudio = false;
        }
      };

      // 区間ごとに新しい読み取り器を使う（2本目以降で止まる端末への対策）
      diag.step = "映像の読み取り";
      const vSink = new VideoSampleSink(vt);
      for await (const smp of vSink.samples(a, b)) {
        const t = offset + Math.max(0, smp.timestamp - a);
        if (t >= offset + len) { smp.close(); break; }
        if (asrc) await pushAudioUntil(t);
        smp.setTimestamp(t);
        smp.setDuration(Math.min(smp.duration || 1 / fps, offset + len - t));
        diag.step = "映像の作り直し";
        await vsrc.add(smp);
        smp.close();
        onProgress((t + (smp.duration || 0)) / total);
      }
      if (asrc) await pushAudioUntil(Infinity);
      offset += len;
      onProgress(offset / total);
    }

    diag.step = "MP4 の組み立て";
    await output.finalize();
    return new Blob([output.target.buffer], { type: "video/mp4" });
  } catch (err) {
    const d = Object.entries(diag).map(([k, v]) => k + "=" + v).join(" ");
    const e = new Error((err && err.message) || String(err));
    e.diag = d;
    throw e;
  } finally {
    try { input.dispose && input.dispose(); } catch (e) {}
  }
}
