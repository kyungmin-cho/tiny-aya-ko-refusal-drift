"""E5 포드를 지우기 전 필수 단계(맥, 표준 라이브러리 + ssh/rsync): 공개 가중치만 받고 sha256을 포드·맥에서 각각 계산해 맞춘다.

받는 것: 포드 <pod_root>/release/ 아래 C100_ep3_adapter/와 규칙으로 고른 T100_ep3_resta<α>/ 뿐(RELEASE_SHA256.json에 적힌 파일만).
  T100 어댑터·다른 α는 받지 않는다(목록에 있으면 멈춘다).
순서
  1) 포드에 RELEASE_READY가 없으면 멈춤 → 맥 <mac_root>/DO_NOT_DELETE_POD(이유)
  2) 포드 RELEASE_SHA256.json을 읽고, 포드에서 sha256sum을 다시 계산해 목록과 맞춘다(포드 계산)
  3) rsync로 <dest>/(프로젝트 밖 <weights_dir>)에 받는다
  4) 맥에서 sha256을 계산해 포드 값과 맞춘다(맥 계산)
  5) 모두 같으면 <dest>/WEIGHTS_OK·<mac_root>/WEIGHTS_OK를 쓰고, 포드에 WEIGHTS_OK를 쓰고 포드의 DO_NOT_DELETE_POD를 지운다.
     하나라도 다르면 <mac_root>/DO_NOT_DELETE_POD에 불일치 목록을 쓰고 종료 코드 1(포드를 지우면 안 됨)
--results: 가중치가 아닌 결과(eval·train 메타·state·agg·samples, 포드 logs는 logs_pod/)를 포드 파일 목록대로 받고 sha256을 맞춘다(*.safetensors 제외).
  결과 받기가 실패하면 가중치가 맞아도 WEIGHTS_OK를 쓰되 DO_NOT_DELETE_POD는 풀지 않고 종료 코드 1.
사용: python3 -I -B e5_mac_fetch.py --ssh "ssh -i KEY -o IdentitiesOnly=yes -o BatchMode=yes -p PORT root@HOST" \
        --pod-root <pod_repo>/experiments/e5 --mac-root <PROJ>/experiments/e5 --dest <weights_dir> [--results]
시험: --local-pod-root <dir>를 주면 ssh 대신 같은 맥의 폴더를 포드로 본다(tests/test_e5.py).
"""
import argparse, datetime, hashlib, json, os, shlex, shutil, subprocess, sys

KST = datetime.timezone(datetime.timedelta(hours=9))
now = lambda: datetime.datetime.now(KST).isoformat(timespec="seconds")
PULL_WORKERS = 6  # 동시 rsync 수(E5 본실행에서 추가)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Pod:
    def __init__(self, ssh=None, local_root=None, pod_root=None):
        self.ssh, self.local, self.root = ssh, local_root, pod_root

    def path(self, rel):
        return os.path.join(self.local or self.root, rel)

    def run(self, cmd):
        if self.local:
            return subprocess.run(["bash", "-c", cmd.replace("{ROOT}", shlex.quote(self.local))], capture_output=True, text=True)
        return subprocess.run(shlex.split(self.ssh) + [cmd.replace("{ROOT}", shlex.quote(self.root))], capture_output=True, text=True)

    def read(self, rel):
        r = self.run(f"cat {{ROOT}}/{shlex.quote(rel)}")
        return r.stdout if r.returncode == 0 else None

    def sha256(self, rels):
        q = " ".join(shlex.quote(x) for x in rels)
        r = self.run(f"cd {{ROOT}} && sha256sum {q} 2>/dev/null || shasum -a 256 {q}")
        out = {}
        for line in r.stdout.splitlines():
            h, p = line.split(None, 1)
            out[p.strip().lstrip("*")] = h
        return out

    def pull(self, rel_dir, dest, excludes=()):
        os.makedirs(dest, exist_ok=True)
        if self.local:
            src = os.path.join(self.local, rel_dir)
            for dp, _, fs in os.walk(src):
                for f in fs:
                    rp = os.path.relpath(os.path.join(dp, f), src)
                    if any(_match(rp, e) for e in excludes):
                        continue
                    os.makedirs(os.path.join(dest, os.path.dirname(rp)), exist_ok=True)
                    shutil.copy2(os.path.join(dp, f), os.path.join(dest, rp))
            return 0
        host = shlex.split(self.ssh)[-1]
        ssh_e = " ".join(shlex.split(self.ssh)[:-1])
        # E5 본실행에서 바꿈: 한 연결이 약 3MB/s로 막혀(예행 측정) 파일을 크기순으로 PULL_WORKERS개 묶음에 나눠 rsync를 동시에 돌린다.
        r = self.run(f"cd {{ROOT}}/{shlex.quote(rel_dir)} && find . -type f -printf '%s %P\\n'")
        if r.returncode != 0:
            return r.returncode
        items = []
        for line in r.stdout.splitlines():
            sz, p = line.split(" ", 1)
            if not any(_match(p, e) for e in excludes):
                items.append((int(sz), p))
        if not items:
            return 0
        groups = [[0, []] for _ in range(min(PULL_WORKERS, len(items)))]
        for sz, p in sorted(items, reverse=True):
            g = min(groups, key=lambda x: x[0]); g[0] += sz; g[1].append(p)
        import tempfile
        procs = []
        for _, ps in groups:
            lf = tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False); lf.write("\n".join(ps) + "\n"); lf.close()
            cmd = ["rsync", "-a", "--partial", "--files-from", lf.name, "-e", ssh_e, f"{host}:{self.root}/{rel_dir}/", dest + "/"]
            procs.append((subprocess.Popen(cmd), lf.name))
        rcs = []
        for pr, ln in procs:
            rcs.append(pr.wait()); os.remove(ln)
        return next((x for x in rcs if x != 0), 0)


def _match(rp, pat):
    import fnmatch
    return fnmatch.fnmatch(rp, pat) or any(fnmatch.fnmatch(part, pat) for part in rp.split(os.sep))


def fail(mac_root, why):
    os.makedirs(mac_root, exist_ok=True)
    with open(os.path.join(mac_root, "DO_NOT_DELETE_POD"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kst=now(), why=why), ensure_ascii=False) + "\n")
    print(json.dumps(dict(ok=False, why=why), ensure_ascii=False))
    return 1


def fetch_weights(pod, mac_root, dest, clear_marker=True):
    if pod.read("RELEASE_READY") is None:
        return fail(mac_root, "포드에 RELEASE_READY 없음(오케스트레이터가 공개 후보를 다 묶지 못함)")
    man = json.loads(pod.read("release/RELEASE_SHA256.json") or "{}")
    files = man.get("files", {})
    if not man.get("complete"):
        return fail(mac_root, "RELEASE_SHA256.json complete=false(C100 어댑터나 고른 ReSta가 빠짐)")
    if not files:
        return fail(mac_root, "RELEASE_SHA256.json이 비었거나 없음")
    sel_tag = (man.get("selection") or {}).get("selected_tag")
    bad = [k for k in files if not (k.startswith("C100_ep3_adapter/") or (sel_tag and k.startswith(sel_tag + "/")))]
    if bad:
        return fail(mac_root, f"공개 목록 밖 파일: {bad[:5]}")
    pod_sha = pod.sha256([f"release/{k}" for k in files])
    pod_mis = [k for k in files if pod_sha.get(f"release/{k}") != files[k]["sha256"]]
    if pod_mis:
        return fail(mac_root, f"포드에서 다시 계산한 sha256이 목록과 다름: {pod_mis}")
    rc = pod.pull("release", dest, excludes=("RELEASE_SHA256.json",))
    if rc != 0:
        return fail(mac_root, f"rsync 실패 rc={rc}")
    mac_sha = {k: sha256_file(os.path.join(dest, k)) if os.path.exists(os.path.join(dest, k)) else None for k in files}
    mis = [k for k in files if mac_sha[k] != pod_sha[f"release/{k}"]]
    extra = []
    for dp, _, fs in os.walk(dest):
        for f in fs:
            rp = os.path.relpath(os.path.join(dp, f), dest)
            if rp not in files and rp not in ("WEIGHTS_OK", "RELEASE_SHA256.json"):
                extra.append(rp)
    if mis or extra:
        return fail(mac_root, dict(sha_mismatch=mis, unexpected_files=extra))
    json.dump(man, open(os.path.join(dest, "RELEASE_SHA256.json"), "w"), ensure_ascii=False, indent=1)
    ok = dict(kst=now(), files={k: dict(pod_sha256=pod_sha[f"release/{k}"], mac_sha256=mac_sha[k], bytes=files[k]["bytes"]) for k in files},
              selected=man.get("selection", {}).get("selected"), fallback=man.get("selection", {}).get("fallback"), dest=dest)
    os.makedirs(mac_root, exist_ok=True)
    for p in (os.path.join(dest, "WEIGHTS_OK"), os.path.join(mac_root, "WEIGHTS_OK")):
        json.dump(ok, open(p, "w"), ensure_ascii=False, indent=1)
    if not clear_marker:  # 결과 받기가 실패했으면 가중치가 맞아도 포드 지우기 금지 표시를 풀지 않는다
        print(json.dumps(dict(ok=True, weights_only=True, note="결과 받기 실패로 DO_NOT_DELETE_POD 유지"), ensure_ascii=False))
        return 1
    if os.path.exists(os.path.join(mac_root, "DO_NOT_DELETE_POD")):
        os.replace(os.path.join(mac_root, "DO_NOT_DELETE_POD"), os.path.join(mac_root, f"DO_NOT_DELETE_POD.resolved-{now()[:19].replace(':', '')}"))
    pod.run("date -Iseconds > {ROOT}/WEIGHTS_OK && rm -f {ROOT}/DO_NOT_DELETE_POD")
    print(json.dumps(dict(ok=True, n_files=len(files), selected=ok["selected"], fallback=ok["fallback"]), ensure_ascii=False))
    return 0


RESULT_SUBDIRS = {"eval": "eval", "train": "train", "state": "state", "agg": "agg", "samples": "samples", "logs": "logs_pod"}  # 포드 logs는 맥 logs와 이름이 겹쳐 logs_pod/로 받음
RESULT_EXCLUDES = ("*.safetensors", "*.bin", "*.pt", "_trainer_tmp", "safetune_results", "_ak_state")


def fetch_results(pod, mac_root):
    """포드에 있는 결과 파일 목록을 포드에서 만들고, 그 파일만 받아 포드·맥 sha256을 맞춘다(맥에만 있는 파일은 비교하지 않음)."""
    rels = []
    for sub, dst in RESULT_SUBDIRS.items():
        r = pod.run(f"cd {{ROOT}} && [ -d {sub} ] && find {sub} -type f || true")
        files = [x for x in r.stdout.splitlines() if x and not any(_match(x, e) for e in RESULT_EXCLUDES)]
        if not files:
            continue
        rc = pod.pull(sub, os.path.join(mac_root, dst), excludes=RESULT_EXCLUDES)
        if rc != 0:
            return fail(mac_root, f"결과 rsync 실패 {sub} rc={rc}")
        rels += [(f, os.path.join(dst, os.path.relpath(f, sub))) for f in files]
    ps = pod.sha256([f for f, _ in rels])
    mis = [f for f, m in rels if not os.path.exists(os.path.join(mac_root, m)) or ps.get(f) != sha256_file(os.path.join(mac_root, m))]
    print(json.dumps(dict(results_files=len(rels), mismatch=len(mis)), ensure_ascii=False))
    return 0 if not mis else fail(mac_root, dict(results_sha_mismatch=mis[:20]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh"); ap.add_argument("--pod-root", default=None, help="experiments/e5 folder of the repository on the GPU machine")
    ap.add_argument("--local-pod-root", default=None)
    ap.add_argument("--mac-root", required=True); ap.add_argument("--dest", required=True, help="weights folder outside the project, <weights_dir>")
    ap.add_argument("--results", action="store_true")
    a = ap.parse_args()
    assert a.local_pod_root or (a.ssh and a.pod_root), "--ssh needs --pod-root"
    proj_like = os.path.abspath(a.dest).startswith(os.path.abspath(os.path.join(a.mac_root, "..", "..")) + os.sep)
    if proj_like and not a.local_pod_root:
        sys.exit(fail(a.mac_root, "dest가 프로젝트 폴더 안임(가중치는 프로젝트 밖 <weights_dir>에 둔다)"))
    pod = Pod(a.ssh, a.local_pod_root, a.pod_root)
    rc = fetch_results(pod, a.mac_root) if a.results else 0
    rc = fetch_weights(pod, a.mac_root, a.dest, clear_marker=(rc == 0)) or rc
    sys.exit(rc)


if __name__ == "__main__":
    main()
