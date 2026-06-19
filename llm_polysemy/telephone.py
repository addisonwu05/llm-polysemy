"""Telephone-game experiment: iterated image<->text sense drift, family-wise.

Each family runs its best image model <-> its best text (captioner) model:
  word -> [image] -> img0 -> [caption: "describe in one sentence"] -> cap0
       -> [image] -> img1 -> [caption] -> cap1 -> ... (N_ITERS image generations)
30 chains per (family, word); seed words = that image model's own top-entropy words.

Phases (python -m llm_polysemy.telephone --phase {generate,classify,histograms,all}):
  generate  : produce images+captions, resumable (chain-level).
  classify  : both judges label every image AND caption into the word's senses.
  histograms: aggregate per (family, word, iter) + write entropy-vs-iteration curves.

Layout: data/telephone/<family>/<word>/c<NN>/i<t>.{png,txt}
        data/telephone/classifications.json, histograms.json, curves.json
"""
import argparse, asyncio, base64, json, math
from pathlib import Path
from dotenv import load_dotenv

from .keys import build_pools
from .providers import openai_sdk
from . import schemas
from .meanings import load_meanings
from .config import JUDGES, GEN_PER_KEY_CONCURRENCY, PER_KEY_CONCURRENCY

ROOT = Path("data/telephone")
MEAN = load_meanings("data/meanings.json")
N_CHAINS, N_ITERS = 30, 5
CAPTION_PROMPT = "Describe this image in one sentence."

FAMILIES = {
    "openai": dict(img_api="gpt-image-2", cap_api="gpt-5.5", sdk="openai", base_url=None,
                   key_prefix="openai_api_key", words=["button", "bark", "trunk", "post", "pitch"]),
    "gemini": dict(img_api="gemini-3-pro-image-preview", cap_api="gemini-3.1-pro-preview", sdk="gemini",
                   base_url=None, key_prefix="gemini_api_key",
                   words=["python", "jordan", "shot", "crane", "batter"]),
    "grok":   dict(img_api="grok-imagine-image-quality", cap_api="grok-4.3", sdk="openai",
                   base_url="https://api.x.ai/v1", key_prefix="xai_api_key",
                   words=["pitcher", "match", "capital", "mortar", "will"]),
}

# ── generation primitives (per-family dispatch) ─────────────────────────────────

def gen_image(fam, prompt, pool):
    c = FAMILIES[fam]
    if c["sdk"] == "gemini":
        from google import genai
        from google.genai import types
        cli = genai.Client(api_key=pool.next())
        r = cli.models.generate_content(model=c["img_api"], contents=f"Generate an image: {prompt}",
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]))
        for p in (r.candidates[0].content.parts or []):
            if p.inline_data is not None:
                return p.inline_data.data
        raise ValueError("no image returned")
    if fam == "openai":
        return openai_sdk.generate_image_openai(prompt, c["img_api"], pool)
    return openai_sdk.generate_image_grok(prompt, c["img_api"], pool, c["base_url"])  # grok

def caption(fam, img, pool):
    c = FAMILIES[fam]
    if c["sdk"] == "gemini":
        from google import genai
        from google.genai import types
        cli = genai.Client(api_key=pool.next())
        r = cli.models.generate_content(model=c["cap_api"],
            contents=[types.Part.from_bytes(data=img, mime_type="image/png"), CAPTION_PROMPT])
        return (r.text or "").strip()
    cli = openai_sdk._client(pool.next(), c["base_url"])
    b64 = base64.b64encode(img).decode()
    resp = cli.chat.completions.create(model=c["cap_api"], messages=[{"role": "user", "content": [
        {"type": "text", "text": CAPTION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
    return (resp.choices[0].message.content or "").strip()

async def _retry(fn, tries=3):
    for a in range(tries):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            if a == tries - 1:
                raise
            await asyncio.sleep(2 * (a + 1))

# ── generation phase ────────────────────────────────────────────────────────────

def _cdir(fam, word, ci): return ROOT / fam / word / f"c{ci:02d}"

def _last_done(fam, word, ci):
    d = _cdir(fam, word, ci); last = -1
    for t in range(N_ITERS):
        if (d / f"i{t}.png").exists() and (d / f"i{t}.txt").exists():
            last = t
        else:
            break
    return last

async def _run_chain(fam, word, ci, pool, sem, bar):
    d = _cdir(fam, word, ci); d.mkdir(parents=True, exist_ok=True)
    last = _last_done(fam, word, ci)
    prompt = word if last < 0 else (d / f"i{last}.txt").read_text().strip()
    done_pre = last + 1
    for t in range(done_pre, N_ITERS):
        async with sem:
            try:
                img = await _retry(lambda: gen_image(fam, prompt, pool))
                (d / f"i{t}.png").write_bytes(img)
                cap = await _retry(lambda: caption(fam, img, pool))
                (d / f"i{t}.txt").write_text(cap)
                prompt = cap
            except Exception as e:
                print(f"  [err] {fam}/{word}/c{ci:02d}/i{t}: {str(e)[:80]}", flush=True)
                break
    bar["n"] += 1
    if bar["n"] % 25 == 0:
        print(f"  {fam}: {bar['n']}/{bar['tot']} chains done", flush=True)

async def generate(families):
    # Families use separate API providers (OpenAI / Gemini / xAI), so run them all
    # CONCURRENTLY — each chain still bounded by its own family's key-pool semaphore.
    tasks = []
    for fam in families:
        c = FAMILIES[fam]
        pool = build_pools({fam: c["key_prefix"]})[fam]
        sem = asyncio.Semaphore(len(pool) * GEN_PER_KEY_CONCURRENCY)
        tot = len(c["words"]) * N_CHAINS
        bar = {"n": 0, "tot": tot}
        print(f"=== GENERATE {fam}: {c['img_api']} <-> {c['cap_api']} | {tot} chains x {N_ITERS} iters (families parallel) ===", flush=True)
        tasks += [_run_chain(fam, w, ci, pool, sem, bar) for w in c["words"] for ci in range(N_CHAINS)]
    await asyncio.gather(*tasks)
    print("  all-family generation complete", flush=True)

# ── classification phase ─────────────────────────────────────────────────────────

CLS_FILE = ROOT / "classifications.json"

def _opts(word):
    L = [f'- "{s["sense"]}": {s["meaning"]}' for s in MEAN[word]]
    L += ['- "multiple": the image/description clearly shows two or more of the above together',
          '- "unclear": you cannot tell, or it matches none of the above',
          '- "other": it clearly shows a different, specific meaning not listed above']
    return "\n".join(L)

def _img_prompt(word):
    return (f'Look at this image and decide which meaning of the word "{word}" it depicts.\n\n'
            f'Choose exactly one of these labels:\n{_opts(word)}\n\nRespond with only the label.')

def _cap_prompt(word, cap):
    return (f'Here is a description of an image: "{cap}"\nDecide which meaning of the word "{word}" '
            f'the described scene corresponds to.\n\nChoose exactly one of these labels:\n{_opts(word)}\n\n'
            f'Respond with only the label.')

def _judge(judge_name, prompt, labels, pool, image=None):
    if judge_name == "gpt":
        sch = schemas.single_schema_gpt(labels)
        lab = openai_sdk.judge(JUDGES["gpt"], prompt, sch, pool, image=image)["label"]
    else:
        from .providers import google_genai
        sch = schemas.single_schema_gemini(labels)
        lab = google_genai.judge(JUDGES["gemini"], prompt, sch, pool, image=image)["label"]
    return lab if lab in labels else "unclear"

async def classify(families):
    load_dotenv()
    pools = build_pools({"gpt": "openai_api_key", "gemini": "gemini_api_key"})
    sems = {j: asyncio.Semaphore(len(pools[j]) * PER_KEY_CONCURRENCY) for j in ("gpt", "gemini")}
    results = json.loads(CLS_FILE.read_text()) if CLS_FILE.exists() else {}
    lock = asyncio.Lock(); done = [0]

    async def cls_one(fam, word, ci, t):
        d = _cdir(fam, word, ci)
        png, txt = d / f"i{t}.png", d / f"i{t}.txt"
        if not (png.exists() and txt.exists()):
            return
        node = results.setdefault(fam, {}).setdefault(word, {}).setdefault(str(ci), {}).setdefault(str(t), {})
        if node.get("image") and node.get("caption"):
            return
        labels = [s["sense"] for s in MEAN[word]] + ["multiple", "unclear", "other"]
        img = png.read_bytes(); cap = txt.read_text().strip()
        out_img, out_cap = {}, {}
        for j in ("gpt", "gemini"):
            async with sems[j]:
                try:
                    out_img[j] = await _retry(lambda j=j: _judge(j, _img_prompt(word), labels, pools[j], image=img))
                    out_cap[j] = await _retry(lambda j=j: _judge(j, _cap_prompt(word, cap), labels, pools[j]))
                except Exception as e:
                    print(f"  [cls-err] {fam}/{word}/c{ci}/i{t}/{j}: {str(e)[:70]}", flush=True)
        async with lock:
            node["image"] = out_img; node["caption"] = out_cap
            done[0] += 1
            if done[0] % 25 == 0:
                CLS_FILE.write_text(json.dumps(results)); print(f"  classified {done[0]} cells", flush=True)

    tasks = [cls_one(fam, w, ci, t) for fam in families for w in FAMILIES[fam]["words"]
             for ci in range(N_CHAINS) for t in range(N_ITERS)]
    print(f"\n=== CLASSIFY: up to {len(tasks)} productions x2 judges x2 modalities ===", flush=True)
    await asyncio.gather(*tasks)
    CLS_FILE.write_text(json.dumps(results))
    print(f"  classification done ({done[0]} new cells)", flush=True)

# ── histograms + entropy curves ──────────────────────────────────────────────────

def _entropy(counts):
    tot = sum(counts.values())
    if tot == 0:
        return None
    ps = [v / tot for v in counts.values() if v > 0]
    H = -sum(p * math.log2(p) for p in ps)
    k = max(len(counts), 2)
    return H / math.log2(k)   # normalized over observed labels

def build_histograms(families, judge="gpt"):
    results = json.loads(CLS_FILE.read_text())
    hist, curves = {}, {}
    for fam in families:
        for word in FAMILIES[fam]["words"]:
            for t in range(N_ITERS):
                for modality in ("image", "caption"):
                    counts = {}
                    for ci in range(N_CHAINS):
                        lab = results.get(fam, {}).get(word, {}).get(str(ci), {}).get(str(t), {}).get(modality, {}).get(judge)
                        if lab:
                            counts[lab] = counts.get(lab, 0) + 1
                    hist.setdefault(fam, {}).setdefault(word, {}).setdefault(str(t), {})[modality] = counts
                    e = _entropy(counts)
                    if e is not None:
                        curves.setdefault(fam, {}).setdefault(word, {}).setdefault(modality, {})[str(t)] = round(e, 4)
    (ROOT / "histograms.json").write_text(json.dumps(hist, indent=1))
    (ROOT / "curves.json").write_text(json.dumps(curves, indent=1))
    # print collapse curves (image modality)
    print(f"\n=== ENTROPY-vs-ITERATION (image, judge={judge}) ===")
    for fam in families:
        for word in FAMILIES[fam]["words"]:
            c = curves.get(fam, {}).get(word, {}).get("image", {})
            row = " ".join(f"{c.get(str(t), float('nan')):.2f}" for t in range(N_ITERS))
            print(f"  {fam:7s} {word:9s}  {row}")
    print(f"\n  saved -> {ROOT/'histograms.json'}, {ROOT/'curves.json'}")

# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["generate", "classify", "histograms", "all"], default="all")
    p.add_argument("--family", choices=list(FAMILIES) + ["all"], default="all")
    p.add_argument("--judge", choices=list(JUDGES), default="gpt")
    a = p.parse_args()
    fams = list(FAMILIES) if a.family == "all" else [a.family]
    if a.phase in ("generate", "all"):
        asyncio.run(generate(fams))
    if a.phase in ("classify", "all"):
        asyncio.run(classify(fams))
    if a.phase in ("histograms", "all"):
        build_histograms(fams, a.judge)

if __name__ == "__main__":
    main()
