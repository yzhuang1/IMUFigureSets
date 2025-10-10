import os, time
import requests
from pathlib import Path

BASE = "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI"
OUT_DIR = Path("ISRUC_Subgroup1_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"  # 某些服务器会对默认UA挑剔
}

def download_one(subj_id: int, retry=3, pause=2):
    url = f"{BASE}/{subj_id}.rar"
    out = OUT_DIR / f"{subj_id:03d}.rar"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} 已存在")
        return
    for t in range(1, retry+1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk)
            # 简单校验
            if out.stat().st_size == 0 or (total and out.stat().st_size != total):
                raise IOError("文件大小异常")
            print(f"[ok] {out.name}")
            return
        except Exception as e:
            print(f"[{t}/{retry}] {url} 失败: {e}")
            time.sleep(pause)
    print(f"[fail] 放弃 {url}")

if __name__ == "__main__":
    for sid in range(1, 101):   # 1..100
        download_one(sid)
