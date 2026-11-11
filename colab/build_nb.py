# pinpon_cut.ipynb を作る（このファイルを直して `python3 build_nb.py` で作り直す）
import json

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})
def code(t, title=None):
    meta = {"cellView": "form"} if title else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": (("#@title " + title + "\n") if title else "") + t.strip("\n")})

md("""
# 🏓 ピンポンカット（Colab 版）
卓球の試合動画から、ラリーの場面だけをつないだ動画を自動で作ります。

**使い方**：上から順に ▶ を押すだけです（メニューの「ランタイム → すべてのセルを実行」でも可）。

1. 動画を選ぶ
2. 自動でラリーを見つける（グラフで確認できます）
3. 書き出して保存

**判定のしかた**（ブラウザ版と同じ）
- 始まり＝映像：サーブの前は 2 人とも構えて止まる。「止まる → 動き出す」をラリーの始まりにする
- 終わり＝音：その区間の中だけで、最後の打球音を探す
- カメラを固定して撮った動画向けです

動画は Google のサーバー（あなたの Colab の作業場所）に上がります。ブラウザ版と違い、端末の外に出ます。
""")

code("""
VIDEO_PATH = ""  #@param {type:"string"}
#@markdown 空のままなら、ファイルを選ぶボタンが出ます（大きい動画は時間がかかります）。<br>
#@markdown Google ドライブの動画を使うときは、パスを入れてください。例：`/content/drive/MyDrive/match.mp4`
import os
if VIDEO_PATH.startswith("/content/drive") and not os.path.exists("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
if not VIDEO_PATH:
    from google.colab import files
    up = files.upload()
    VIDEO_PATH = next(iter(up))
assert os.path.exists(VIDEO_PATH), "動画が見つかりません：" + VIDEO_PATH
print("動画：", VIDEO_PATH)
""", "1️⃣ 動画を選ぶ")

code('''
import json, subprocess
import numpy as np
from scipy import signal

RATE = 200          # 音の強さを 1 秒に何個とるか
MOTION_FPS = 10     # 映像を 1 秒に何コマ見るか
MOTION_W, MOTION_H, MOTION_DIFF = 160, 90, 12   # 縮めた映像で、明るさが 12 以上変わった画素を「動いた」と数える


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def analyze_video(path):
    """コマごとの「動いた画素の割合」（1 秒に 10 個）"""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={MOTION_FPS},scale={MOTION_W}:{MOTION_H}:flags=area",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, MOTION_H, MOTION_W).astype(np.int16)
    moved = (np.abs(np.diff(frames, axis=0)) > MOTION_DIFF).mean(axis=(1, 2))
    return np.concatenate([[0.0], moved]).astype(np.float32)


def analyze_audio(path):
    """5 ミリ秒ごとの音の強さ。all＝そのまま / mid＝1.5kHz より上 / hi＝5kHz より上"""
    sr = 48000
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32).astype(np.float64)
    if len(x) == 0:
        raise RuntimeError("この動画には音が入っていません")
    def highpass(cut):  # ブラウザ版と同じ、ゆるい高域フィルター
        a = 1 / (1 + 2 * np.pi * cut / sr)
        return signal.lfilter([a, -a], [1, -a], x)
    hop = sr // RATE
    n = len(x) // hop
    energy = lambda y: (y[:n * hop].reshape(n, hop) ** 2).sum(axis=1)
    return {"all": energy(x), "mid": energy(highpass(1500)), "hi": energy(highpass(5000))}


def find_clicks(env, hi_lo_min):
    """打球音（高い音が主体の、短く鋭い音）の時刻。足音は低い音も一緒に鳴るので外れる"""
    hi = 10 * np.log10(env["hi"] + 1e-12)
    lo = 10 * np.log10(np.maximum(1e-12, env["all"] - env["mid"]))
    n = len(hi)
    loud = np.sort(hi)[int(n * 0.9)] + 4          # その動画の中で十分大きい音だけ
    clicks = []
    for i in range(14, n - 17):
        v = hi[i]
        if v < loud: continue
        if hi[i - 16:i].max() > v or hi[i + 1:i + 17].max() >= v: continue        # 前後 80 ミリ秒でいちばん大きい
        if v - np.sort(hi[i - 13:i - 1])[6] < 12: continue                         # 直前より 12dB 以上、急に大きくなった
        if v - lo[i - 1:i + 3].max() < hi_lo_min: continue                         # 低い音が勝っている＝足音など
        clicks.append(i / RATE)
    return np.array(clicks)


def find_stills(motion, still_ratio, still_min):
    """止まっている区間（コマ番号の [始め, 終わり)）と、その判定に使った境目"""
    m = signal.medfilt(np.asarray(motion, np.float64), 3)
    m[0], m[-1] = motion[0], motion[-1]
    limit = np.sort(m)[int(len(m) * 0.75)] * still_ratio
    stills, i, n = [], 0, len(m)
    while i < n:
        if m[i] >= limit: i += 1; continue
        j = i
        while j < n and m[j] < limit: j += 1
        if (j - i) / MOTION_FPS >= still_min: stills.append((i, j))
        i = j
    return stills, limit


def find_rallies(motion, env, duration, hi_lo=-3, gap=1.8, still_ratio=0.2, still_min=0.5,
                 first_click=5.0, pad_before=0.5, pad_after=0.3):
    stills, limit = find_stills(motion, still_ratio, still_min)
    clicks = find_clicks(env, hi_lo)
    rallies = []
    for k, (_, move_at) in enumerate(stills):
        if move_at >= len(motion): continue
        t0 = move_at / MOTION_FPS                                                  # 動き出した時刻
        until = stills[k + 1][0] / MOTION_FPS if k + 1 < len(stills) else duration # 次に止まるまで
        last, hits = None, 0
        for t in clicks:
            if t < t0 - 0.3: continue
            if t > until: break
            if (t - t0 > first_click) if last is None else (t - last > gap): break
            last, hits = t, hits + 1
        if last is None: continue                                                  # 打球音が無い＝ラリーではない
        rallies.append((max(0.0, t0 - pad_before), min(duration, last + pad_after), hits))
    return rallies, stills, clicks, limit

print("準備できました")
''', "2️⃣ 判定の部品を読み込む（中身を見たいときはダブルクリック）")

code('''
SENSITIVITY = "ふつう"  #@param ["少なめ（確実なものだけ）", "ふつう", "多め（取りこぼしを減らす）"]
#@markdown 下の 3 つは細かい調整用です。ふつうは触らなくて大丈夫です。
STILL_RATIO = 0.2   #@param {type:"number"}
STILL_MIN_SEC = 0.5 #@param {type:"number"}
FIRST_CLICK_SEC = 5 #@param {type:"number"}
import matplotlib.pyplot as plt

if "_cache" not in globals() or _cache.get("path") != VIDEO_PATH:   # 同じ動画なら読み取りは 1 回だけ
    print("映像と音を読み取っています…")
    _cache = {"path": VIDEO_PATH, "duration": probe_duration(VIDEO_PATH),
              "motion": analyze_video(VIDEO_PATH), "env": analyze_audio(VIDEO_PATH)}
duration, motion, env = _cache["duration"], _cache["motion"], _cache["env"]

hi_lo, gap = {"少": (-1, 1.5), "ふ": (-3, 1.8), "多": (-5, 2.5)}[SENSITIVITY[0]]
rallies, stills, clicks, limit = find_rallies(motion, env, duration, hi_lo, gap, STILL_RATIO, STILL_MIN_SEC, FIRST_CLICK_SEC)

total = sum(e - s for s, e, _ in rallies)
print(f"{len(rallies)} 本見つけました（合計 {total:.1f} 秒 / 元の動画 {duration:.1f} 秒）")
for i, (s, e, h) in enumerate(rallies, 1):
    print(f"  {i:3d}. {int(s // 60)}:{s % 60:04.1f} → {int(e // 60)}:{e % 60:04.1f}（{e - s:.1f} 秒・打球音 {h} 回）")

# グラフ：黄色＝ラリーと判定 / 灰色＝止まっている / 赤い線＝打球音（文字化けを避けるため英語表記）
t = np.arange(len(motion)) / MOTION_FPS
rows = max(1, int(np.ceil(duration / 60)))
fig, axes = plt.subplots(rows, 1, figsize=(14, 2.2 * rows), squeeze=False)
for r, ax in enumerate(axes[:, 0]):
    a, b = r * 60, (r + 1) * 60
    for s, e, _ in rallies: ax.axvspan(s, e, color="gold", alpha=0.6)
    for i, j in stills: ax.axvspan(i / MOTION_FPS, j / MOTION_FPS, color="gray", alpha=0.25)
    ax.plot(t, motion, color="green", lw=1)
    ax.axhline(limit, color="gray", ls=":", lw=1)
    ax.vlines(clicks, 0, motion.max() * 0.15, color="red", lw=1)
    ax.set_xlim(a, b); ax.set_ylim(0, max(1e-6, float(np.percentile(motion, 99)) * 1.2))
    ax.set_ylabel("motion"); ax.set_xticks(np.arange(a, b + 1, 5))
axes[-1, 0].set_xlabel("time (sec)   yellow = rally / gray = still / red = ball sound")
plt.tight_layout(); plt.show()
''', "3️⃣ 自動でラリーを見つける")

md("""
**グラフの見方**　緑の線＝映像の動きの量、灰色＝止まっていると判定した所、赤い線＝打球音、黄色＝ラリーと判定した区間。

違っていたら、上の「見つける量」を変えて 3️⃣ をもう一度 ▶ してください（読み取りは済んでいるのですぐ終わります）。
手で直したいときは、次のセルの `EDIT` に区間を書けます。
""")

code('''
EDIT = ""  #@param {type:"string"}
#@markdown 空なら自動判定のまま。手で決めるときは「始め-終わり」を秒で、カンマ区切りで書きます。例：`2.7-7.7, 10.9-14.5`
LEAD_SEC = 0.0  #@param {type:"number"}
TAIL_SEC = 1.2  #@param {type:"number"}
MAX_SIDE = 1920 #@param [1920, 1280, 960, 0] {type:"raw"}
#@markdown `MAX_SIDE`：長い辺の上限（0 で元のまま）
from pathlib import Path

if EDIT.strip():
    marks = [tuple(float(v) for v in part.split("-")) for part in EDIT.split(",") if part.strip()]
else:
    marks = [(s, e) for s, e, _ in rallies]
assert marks, "区間がありません"

# 前後の余白を足し、重なったものは 1 つにまとめる
segs = []
for s, e in sorted(marks):
    s, e = max(0.0, s - LEAD_SEC), min(duration, e + TAIL_SEC)
    if segs and s <= segs[-1][1]: segs[-1][1] = max(segs[-1][1], e)
    else: segs.append([s, e])

has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", VIDEO_PATH],
                                capture_output=True, text=True).stdout.strip())
parts, labels = [], ""
for i, (s, e) in enumerate(segs):
    parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
    if has_audio: parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
    labels += f"[v{i}]" + (f"[a{i}]" if has_audio else "")
scale = f",scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'" if MAX_SIDE else ""
parts.append(f"{labels}concat=n={len(segs)}:v=1:a={1 if has_audio else 0}[vc]" + ("[outa]" if has_audio else ""))
parts.append(f"[vc]format=yuv420p{scale}[outv]")
Path("filter.txt").write_text(";\\n".join(parts))

OUT_PATH = str(Path(VIDEO_PATH).with_name(Path(VIDEO_PATH).stem + "_rally.mp4"))
cmd = ["ffmpeg", "-y", "-v", "error", "-stats", "-i", VIDEO_PATH, "-filter_complex_script", "filter.txt", "-map", "[outv]"]
if has_audio: cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "160k"]
cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", OUT_PATH]
print(f"{len(segs)} 本・合計 {sum(e - s for s, e in segs):.1f} 秒を書き出しています…")
subprocess.run(cmd, check=True)
print("できました：", OUT_PATH, f"（{os.path.getsize(OUT_PATH) / 1048576:.1f}MB）")
''', "4️⃣ ラリーだけの動画を書き出す")

code('''
from google.colab import files
files.download(OUT_PATH)
''', "5️⃣ 保存する（ダウンロード）")

md("""
---
### うまく切れなかったとき
3️⃣ のグラフの画像（右クリック → 画像を保存）と、どこが違ったかを教えてください。基準を合わせ直します。

### この先（GPU を使う改良）
いまの判定は GPU を使っていません（動きの量と音だけなので CPU で足ります）。
GPU が効くのは、選手の姿勢や球を AI で読む方法を足すときです。その場合は「ランタイム → ランタイムのタイプを変更 → GPU」にします。
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
for c in cells: c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open("pinpon_cut.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cells", len(cells))
