"""Measure vLLM-AR throughput via concurrent OCR requests to a local GLM-OCR vLLM endpoint.
For each concurrency level, fire that many requests in flight (vLLM continuous-batches them) and
report aggregate tok/s + req/s. Greedy (temperature 0)."""
import argparse, base64, io, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk

def encode(pil):
    pil=pil.convert("RGB"); buf=io.BytesIO(); pil.save(buf,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def req(port, du, prompt, mx):
    body={"model":"glm-ocr","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":du}},{"type":"text","text":prompt}]}],
        "max_tokens":mx,"temperature":0.0}
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=600) as resp: o=json.loads(resp.read())
    return o["usage"]["completion_tokens"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=8200)
    ap.add_argument("--max_tokens",type=int,default=256); ap.add_argument("--concurrency",default="1,8,16,32")
    ap.add_argument("--n",type=int,default=19)
    a=ap.parse_args()
    man=json.load(open("PRESENTATION/reviz/english_crops.json"))[:a.n]; ds=load_from_disk("./olmocr_test_ds/hf_dataset")
    crops=[(encode(ds[p["idx"]]["image"]),ds[p["idx"]]["prompt"]) for p in man]
    req(a.port,crops[0][0],crops[0][1],16)  # warmup
    print("[vllm-thrpt] port",a.port,flush=True)
    for C in [int(x) for x in a.concurrency.split(",")]:
        batch=[crops[i%len(crops)] for i in range(C)]
        t0=time.time()
        with ThreadPoolExecutor(max_workers=C) as ex:
            toks=list(ex.map(lambda c:req(a.port,c[0],c[1],a.max_tokens),batch))
        dt=time.time()-t0; tt=sum(toks)
        print(f"[vllm-thrpt] concurrency={C}: {tt/dt:.0f} tok/s, {C/dt:.1f} req/s ({tt}tok {dt:.1f}s)",flush=True)
    print("VLLM_THRPT_DONE",flush=True)

if __name__=="__main__": main()
