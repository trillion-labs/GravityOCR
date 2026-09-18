"""UniMER CDM scoring (formula recognition axis).


Why CDM rather than BLEU: BLEU / edit distance mistake LaTeX notation dialects for recognition errors
(the same prediction can score poorly on BLEU and near-perfectly on CDM). CDM compares rendered
formulas, so it is the axis reported.

Recipe (keep fixed so runs stay comparable):
    GT   = `gt` field of `<REF>/<sub>.jsonl` (a reference run defines the image set)
    pred = `pred` field of `<PREDS>/<sub>.jsonl`
  CDM  = OmniDocBench-eval `src.metrics.cdm_metric.CDM.evaluate(gt,pred).F1_score`
    dedup by md5 of (gt, pred); 30 s SIGALRM per pair; render failures are EXCLUDED from the mean, not scored 0
    overall = unweighted mean of the 4 subset means

Run (CPU only, ~16 min per model, no GPU):
  cd ${DOCR_WORKSPACE}/OmniDocBench-eval && \
  PATH=${DOCR_WORKSPACE}/cdm_render/env/bin:$PATH \
  CDM_TEXLIVE_BIN=${DOCR_WORKSPACE}/cdm_render/texlive/bin/x86_64-linux \
  CDM_SAVE_VIS=0 W=64 WR=16 \
    PREDS=<predictions dir> LABEL=<tag> \
  .venv-score/bin/python ${DOCR_ROOT}/tools/unimer_cdm.py

    Without the render environment (PATH + CDM_TEXLIVE_BIN) CDM silently becomes 0.
    CDM_SAVE_VIS=0 only disables the visualization PNGs (same F1, saves ~10 GB of disk).

Render flakiness: a batch render timeout used to produce spurious zeros. `cdm_metric.py` now retries
(`CDM_RETRY=3`); this script still writes both raw and repaired scores, so report both if they differ.

MinerU2.5-Pro 0.9557 · MinerU-Diff 0.9515 · Hunyuan 0.9417 · dots 0.9100 · DS2 0.8630 · Paddle 0.7850
"""
import sys, os, json, statistics, signal, hashlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, '${DOCR_WORKSPACE}/OmniDocBench-eval')
BM = "${DOCR_WORKSPACE}/benchmarks"
PREDS = os.environ.get("PREDS", f"{BM}/unimer_run/preds_gt40000")
LABEL = os.environ.get("LABEL", "gt40000")
OUT_PAIRS = os.environ.get("OUT_PAIRS", "/tmp/unimer_cdm_gt40000_pairs.json")
SUBS = ["spe", "cpe", "sce", "hwe"]
TIMEOUT = int(os.environ.get("TMO", "30"))

_cdm = None
def _get():
    global _cdm
    if _cdm is None:
        from src.metrics.cdm_metric import CDM
        _cdm = CDM(output_root=f"/tmp/cdmpar_{os.getpid()}")
    return _cdm

def work(args):
    gt, pred, tid, tmo = args
    def h(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, h); signal.alarm(tmo)
    try:
        r = _get().evaluate(gt, pred, img_id=tid)
        return float(r.get("F1_score", 0.0))
    except Exception:
        return None
    finally:
        signal.alarm(0)

def load(d, sub):
    p = f"{d}/{sub}.jsonl"; o = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); o[r.get('image')] = (r.get('gt', ''), r.get('pred', ''))
    return o

base = {sub: load(f"{BM}/unimer_run/preds", sub) for sub in SUBS}
assign = []     # (sub, pairkey)
pair_of = {}    # pairkey -> (gt, pred)
for sub in SUBS:
    imgs = sorted(base[sub].keys())
    m = load(PREDS, sub)
    for im in imgs:
        if im not in m: continue
        gt = base[sub][im][0] or ""; pred = m[im][1] or ""
        k = hashlib.md5((gt + "\x00" + pred).encode()).hexdigest()
        pair_of.setdefault(k, (gt, pred)); assign.append((sub, k))
uniq = list(pair_of.items())
print(f"[{LABEL}] assignments={len(assign)}  unique(gt,pred)={len(uniq)}", flush=True)

pairres = {}
t0 = time.time()
with ProcessPoolExecutor(max_workers=int(os.environ.get("W", "64"))) as ex:
    futs = {ex.submit(work, (gp[0], gp[1], k, TIMEOUT)): k for k, gp in uniq}
    done = 0
    for fu in as_completed(futs):
        k = futs[fu]
        try: pairres[k] = fu.result()
        except Exception: pairres[k] = None
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(uniq)}  {time.time()-t0:.0f}s", flush=True)
print(f"[{LABEL}] pass1 done {time.time()-t0:.0f}s", flush=True)

# ---- flake probe: retry every None / exact-0 pair, serially, with a long timeout ----
susp = [k for k, v in pairres.items() if v is None or v == 0.0]
print(f"[{LABEL}] suspicious (None or F1==0): {len(susp)} of {len(uniq)} — retrying serially", flush=True)
retry = {}
t1 = time.time()
with ProcessPoolExecutor(max_workers=int(os.environ.get("WR", "8"))) as ex:
    futs = {ex.submit(work, (pair_of[k][0], pair_of[k][1], k, int(os.environ.get("TMO_RETRY", "120")))): k for k in susp}
    done = 0
    for fu in as_completed(futs):
        k = futs[fu]
        try: retry[k] = fu.result()
        except Exception: retry[k] = None
        done += 1
        if done % 200 == 0:
            print(f"  retry {done}/{len(susp)}  {time.time()-t1:.0f}s", flush=True)
print(f"[{LABEL}] retry done {time.time()-t1:.0f}s", flush=True)

repaired = dict(pairres)
n_fixed = 0
for k, v in retry.items():
    if v is not None and (pairres.get(k) is None or v > (pairres.get(k) or 0.0)):
        repaired[k] = v
        if v > 0: n_fixed += 1

json.dump({"pair_of_keys": None, "pass1": pairres, "retry": retry,
           "assign": assign}, open(OUT_PAIRS, "w"))

def agg(store):
    per = {}
    for sub in SUBS:
        vs = [store.get(k) for s, k in assign if s == sub]
        vs = [v for v in vs if v is not None]
        per[sub] = statistics.mean(vs) if vs else None
        per[sub + "_n"] = len(vs)
    per["n_dropped"] = sum(1 for s, k in assign if store.get(k) is None)
    vals = [per[s] for s in SUBS if per[s] is not None]
    per["avg"] = statistics.mean(vals) if vals else None
    return per

raw = agg(pairres)
rep = agg(repaired)
print(f"\n=== UniMER CDM [{LABEL}] full run ===")
print(f"{'':10s} " + " ".join(f"{s.upper():>7s}" for s in SUBS) + f" {'avg':>7s}")
print(f"{'raw':10s} " + " ".join(f"{raw[s]:7.4f}" for s in SUBS) + f" {raw['avg']:7.4f}")
print(f"{'repaired':10s} " + " ".join(f"{rep[s]:7.4f}" for s in SUBS) + f" {rep['avg']:7.4f}")
print(f"pairs fixed by retry: {n_fixed}")

json.dump({"label": LABEL, "preds_dir": PREDS, "n_assignments": len(assign),
           "n_unique_pairs": len(uniq), "timeout_s": TIMEOUT,
           "raw": raw, "repaired": rep, "n_suspicious": len(susp), "n_fixed_by_retry": n_fixed},
          open(f"/tmp/unimer_cdm_{LABEL}.json", "w"), indent=1)
print(f"saved /tmp/unimer_cdm_{LABEL}.json")
