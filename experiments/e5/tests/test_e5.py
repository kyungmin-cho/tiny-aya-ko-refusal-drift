"""E5 단위시험(맥 CPU, GPU 없음). 실행(PROJ에서):
  uv run python -I -B -m unittest discover -s experiments/e5/tests -p 'test_e5.py' -v
"""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
E5 = os.path.dirname(HERE)
PROJ = os.path.dirname(os.path.dirname(E5))
SC = os.path.join(E5, "scripts")
sys.path.insert(0, os.path.join(PROJ, "scripts")); sys.path.insert(0, SC)
import e4_common as C  # noqa: E402
import e5_orchestrate as O  # noqa: E402
import e5_select_release as SR  # noqa: E402

MAN = json.load(open(os.path.join(E5, "data", "data_manifest.json")))
EM = json.load(open(os.path.join(E5, "data", "eval_items_manifest.json")))
E4MAN = json.load(open(os.path.join(PROJ, "experiments/e4/data/data_manifest.json")))


class TestData(unittest.TestCase):
    def test_rows_zero_leak(self):
        for arm in ("T100", "C100"):
            ch = MAN["arms"][arm]["checks"]
            self.assertEqual((ch["rows"], ch["safetune_rows"], ch["zero_label_rows"], ch["leak_rows_present"], ch["empty_responses"]), (2100, 2100, 0, [], 0))
            self.assertEqual(ch["steps_per_epoch"], 525)
        self.assertEqual(MAN["arms"]["T100"]["checks"]["harm_rows"], 100)
        self.assertEqual(MAN["arms"]["C100"]["checks"]["harm_rows"], 0)

    def test_files_sha_and_rows(self):
        for arm in ("T100", "C100"):
            p = os.path.join(E5, "data", MAN["arms"][arm]["file"])
            self.assertEqual(C.sha256_file(p), MAN["arms"][arm]["sha256"])
            rows = C.read_jsonl(p)
            self.assertEqual([r["rid"] for r in rows[:2000]], MAN["arms"][arm]["clean_rids"])
            self.assertFalse(set(r["rid"] for r in rows) & set(C.LEAK_IDS))

    def test_pair_shares_clean2000_and_token_diff(self):
        self.assertEqual(MAN["arms"]["T100"]["clean_rids"], MAN["arms"]["C100"]["clean_rids"])
        self.assertTrue(MAN["pairs"]["T100_vs_C100"]["clean2000_identical_in_order"])
        self.assertLessEqual(abs(MAN["pairs"]["T100_vs_C100"]["label_tokens_diff_ctrl_minus_treat"]), 10)

    def test_harm100_same_as_e4(self):
        self.assertEqual(MAN["arms"]["T100"]["slot_rids"], E4MAN["arms"]["T100"]["slot_rids"])

    def test_new_seed(self):
        self.assertNotEqual(MAN["train_cfg"]["train_seed"], E4MAN["train_cfg"]["train_seed"])

    def test_materialize_reproduces_sha(self):
        d = tempfile.mkdtemp()
        try:
            res = C.materialize(MAN, d)
            self.assertEqual(set(res.values()), {"materialized"})
        finally:
            shutil.rmtree(d)

    def test_eval_items(self):
        for k, n in (("mjen", 315), ("sel", 200), ("mmlu", 600)):
            p = os.path.join(PROJ, EM["files"][k]["path"])
            self.assertEqual(C.sha256_file(p), EM["files"][k]["sha256"])
            self.assertEqual(len(C.read_jsonl(p)), n)
        mm = C.read_jsonl(os.path.join(PROJ, EM["files"]["mmlu"]["path"]))
        rep = {m["gm_idx"] for m in mm if m["split"] == "rep"}; sel = {m["gm_idx"] for m in mm if m["split"] == "sel"}
        self.assertEqual((len(rep), len(sel), len(rep & sel)), (400, 200, 0))
        e1 = C.read_jsonl(os.path.join(PROJ, MAN["eval_inputs"]["e1_items"]["path"]))
        en = C.read_jsonl(os.path.join(PROJ, EM["files"]["mjen"]["path"]))
        self.assertEqual([r["mj_id"] for r in e1], [r["mj_id"] for r in en])
        s = C.read_jsonl(os.path.join(PROJ, EM["files"]["sel"]["path"]))
        self.assertEqual(sum(r["id"].startswith("selh-") for r in s), 100)
        used = set(MAN["arms"]["C100"]["clean_rids"]) | set(MAN["arms"]["C100"]["slot_rids"])
        self.assertFalse({r["rid"] for r in s if r["id"].startswith("selb-")} & used)


class TestTrainSeed(unittest.TestCase):
    def test_wrapper_sets_seed(self):
        import e5_train
        old = C.TRAIN_SEED
        try:
            seed = e5_train.set_seed_from_manifest(E5)
            import e4_train
            self.assertEqual(e4_train.C.TRAIN_SEED, seed)
            self.assertEqual(seed, MAN["train_cfg"]["train_seed"])
        finally:
            C.TRAIN_SEED = old

    def test_seed_reaches_training_args(self):
        """작은 무작위 Cohere2 모델로 PlainSFTTrainer._training_args()의 seed가 새 시드인지(e4_train.main과 같은 호출)."""
        import e5_train, safetune, torch
        from transformers import Cohere2Config, Cohere2ForCausalLM
        from safetune.runner.harden import PlainSFTTrainer
        old = C.TRAIN_SEED
        try:
            seed = e5_train.set_seed_from_manifest(E5)
            safetune.configure(seed=C.TRAIN_SEED, device="cpu")
            cfg = Cohere2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                max_position_embeddings=32, sliding_window=16, pad_token_id=0, bos_token_id=1, eos_token_id=2)
            torch.manual_seed(0)
            tmp = tempfile.mkdtemp()
            tr = PlainSFTTrainer(model=Cohere2ForCausalLM(cfg), tokenizer=None, model_id="tiny", epochs=3, batch_size=4, lr=1e-4, results_dir=tmp)
            args = tr._training_args(os.path.join(tmp, "t"))
            self.assertEqual(args.seed, seed)
            shutil.rmtree(tmp)
        finally:
            C.TRAIN_SEED = old


def make_root():
    tmp = tempfile.mkdtemp(prefix="e5test_")
    root = os.path.join(tmp, "e5"); os.makedirs(os.path.join(root, "data"))
    for f in ("data_manifest.json", "eval_items_manifest.json"):
        shutil.copy(os.path.join(E5, "data", f), os.path.join(root, "data"))
    return tmp, root


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = make_root()
        self._pf = O.preflight
        O.preflight = lambda root: (open(os.path.join(root, "DO_NOT_DELETE_POD"), "a").write("x\n"), {"fake": 1})[1]
        self.runner = [sys.executable, "-I", "-B", os.path.join(HERE, "fake_runner_e5.py")]
        self._cmd = O.command
        # select는 실제 스크립트, agg는 건너뜀(e4_aggregate는 E4 시험에서 확인)

    def tearDown(self):
        O.preflight = self._pf
        shutil.rmtree(self.tmp)

    def _run(self, env=None, **kw):
        old = dict(os.environ)
        os.environ.update(dict(FAKE_CALLS=os.path.join(self.tmp, "calls.txt"), **(env or {})))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return O.run(self.root, self.runner, skip=kw.pop("skip", {"agg"}), **kw)
        finally:
            os.environ.clear(); os.environ.update(old)

    def test_plan(self):
        names = [s["name"] for s in O.plan()]
        self.assertEqual(names[:3], ["preflight", "train_T100", "train_C100"])
        self.assertEqual(sum(1 for s in O.plan() if s["kind"] == "eval"), 7)
        self.assertLess(names.index("eval_T100_ep3_resta1.0"), names.index("select_release"))
        self.assertEqual(names[-1], "auditkit")

    def test_full_run_selects_and_releases(self):
        st = self._run(dict(FAKE_SEL_H="0.25:40,0.5:70,0.75:70,1.0:90", FAKE_SEL_M="0.25:150,0.5:145,0.75:144,1.0:120"))
        self.assertEqual({k for k, v in st.items() if v not in ("done", "skipped")}, set())
        sel = json.load(open(os.path.join(self.root, "agg", "alpha_selection.json")))
        self.assertEqual(sel["selected"], 0.5)  # 1.0은 MMLU 자격 탈락(120 < 150-10), 0.5·0.75 동점 → 작은 α
        self.assertIsNone(sel["fallback"])
        rel = json.load(open(os.path.join(self.root, "release", "RELEASE_SHA256.json")))
        self.assertEqual(set(rel["parts"]), {"C100_ep3_adapter", "T100_ep3_resta0.5"})
        self.assertFalse(any("T100_ep3_adapter" in k or "resta0.25" in k for k in rel["files"]))
        self.assertTrue(os.path.exists(os.path.join(self.root, "RELEASE_READY")))
        self.assertTrue(os.path.exists(os.path.join(self.root, "DO_NOT_DELETE_POD")))
        # 다시 돌리면 학습·평가 호출 없음
        open(os.path.join(self.tmp, "calls.txt"), "w").close()
        self._run()
        self.assertNotIn("e5_train.py", open(os.path.join(self.tmp, "calls.txt")).read())
        self.assertNotIn("e5_eval.py", open(os.path.join(self.tmp, "calls.txt")).read())

    def test_failure_blocks_dependents(self):
        st = self._run(dict(FAKE_FAIL="T100"))
        self.assertEqual(st["train_T100"], "failed")
        self.assertTrue(all(st[f"eval_T100_ep3_resta{a}"] == "blocked" for a in O.ALPHAS))
        self.assertEqual(st["eval_C100_ep3"], "done")
        sel = json.load(open(os.path.join(self.root, "agg", "alpha_selection.json")))
        self.assertEqual(sel["fallback"], "no_selection")
        self.assertFalse(os.path.exists(os.path.join(self.root, "RELEASE_READY")))

    def test_dry_run(self):
        out = O.run(self.root, ["python"], dry=True) if False else None
        with contextlib.redirect_stdout(io.StringIO()):
            out = O.run(self.root, ["uv", "run", "python", "-I", "-B"], dry=True)
        self.assertEqual(len(out["steps"]), len(O.plan()))
        self.assertGreater(out["total_todo_min"], 100)


class TestSelectRule(unittest.TestCase):
    def _mk(self, H, M, B, done=None):
        tmp = tempfile.mkdtemp(); root = os.path.join(tmp, "r")
        def w(tag, h, m, b, saved=True):
            d = os.path.join(root, "eval", tag); os.makedirs(d)
            with open(os.path.join(d, "sel_judge.jsonl"), "w") as f:
                for i in range(100):
                    f.write(json.dumps(dict(id=f"selh-{i}", refusal=i < h)) + "\n"); f.write(json.dumps(dict(id=f"selb-{i}", refusal=i < b)) + "\n")
            with open(os.path.join(d, "mmlu.jsonl"), "w") as f:
                for i in range(200):
                    f.write(json.dumps(dict(id=f"m{i}", split="sel", correct=i < m)) + "\n")
            open(os.path.join(d, "DONE.json"), "w").write("{}")
            if saved:
                open(os.path.join(d, "SAVED_OK"), "w").write("x")
        w("base", 90, 150, 5)
        for al in SR.ALPHAS:
            if done is None or al in done:
                w(SR.tag_of(al), H[al], M[al], B[al])
        return tmp, root

    def test_tie_smaller_alpha(self):
        tmp, root = self._mk({.25: 60, .5: 60, .75: 50, 1.0: 40}, {.25: 150, .5: 150, .75: 150, 1.0: 150}, {.25: 5, .5: 5, .75: 5, 1.0: 5})
        self.assertEqual(SR.select(root)["selected"], 0.25); shutil.rmtree(tmp)

    def test_overrefusal_excludes(self):
        tmp, root = self._mk({.25: 60, .5: 80, .75: 90, 1.0: 95}, {a: 150 for a in SR.ALPHAS}, {.25: 5, .5: 15, .75: 16, 1.0: 30})
        r = SR.select(root)
        self.assertEqual(r["selected"], 0.5)  # B ≤ 5+10 = 15 → 0.25·0.5만 자격
        shutil.rmtree(tmp)

    def test_no_eligible_fallback(self):
        tmp, root = self._mk({a: 80 for a in SR.ALPHAS}, {.25: 130, .5: 135, .75: 120, 1.0: 100}, {a: 5 for a in SR.ALPHAS})
        r = SR.select(root)
        self.assertEqual((r["selected"], r["fallback"]), (0.5, "no_eligible")); shutil.rmtree(tmp)

    def test_mmlu_near_chance_drops_m(self):
        tmp, root = self._mk({.25: 60, .5: 70, .75: 80, 1.0: 90}, {a: 50 for a in SR.ALPHAS}, {a: 5 for a in SR.ALPHAS})
        # base M을 55(우연 근처)로 덮어씀
        with open(os.path.join(root, "eval", "base", "mmlu.jsonl"), "w") as f:
            for i in range(200):
                f.write(json.dumps(dict(id=f"m{i}", split="sel", correct=i < 55)) + "\n")
        r = SR.select(root)
        self.assertEqual((r["selected"], r["m_criterion"]), (1.0, "dropped_near_chance")); shutil.rmtree(tmp)

    def test_missing_alpha_ignored(self):
        tmp, root = self._mk({.25: 60, .5: 99, .75: 70, 1.0: 70}, {a: 150 for a in SR.ALPHAS}, {a: 5 for a in SR.ALPHAS}, done={0.25, 0.75})
        r = SR.select(root)
        self.assertEqual(r["selected"], 0.75); self.assertEqual(r["missing"], [0.5, 1.0]); shutil.rmtree(tmp)


class TestMacFetch(unittest.TestCase):
    def _pod(self, corrupt=False, extra_t100=False):
        tmp = tempfile.mkdtemp(prefix="e5fetch_")
        pod = os.path.join(tmp, "pod"); os.makedirs(pod)
        for p, b in (("train/C100/ep3/adapter_model.safetensors", b"c100"), ("train/C100/ep3/adapter_config.json", b"{}"),
                     ("release_cand/T100_ep3_resta0.5/model.safetensors", b"merged"), ("release_cand/T100_ep3_resta0.5/config.json", b"{}")):
            os.makedirs(os.path.dirname(os.path.join(pod, p)), exist_ok=True); open(os.path.join(pod, p), "wb").write(b)
        open(os.path.join(pod, "DO_NOT_DELETE_POD"), "w").write("x")
        SR.build_release(pod, dict(selected=0.5, selected_tag="T100_ep3_resta0.5"))
        if corrupt:  # 목록을 만든 뒤 포드 파일이 바뀐 경우
            p = os.path.join(pod, "release", "T100_ep3_resta0.5", "model.safetensors"); os.remove(p); open(p, "wb").write(b"broken")
        if extra_t100:
            os.makedirs(os.path.join(pod, "release", "T100_ep3_adapter"))
            m = json.load(open(os.path.join(pod, "release", "RELEASE_SHA256.json")))
            m["files"]["T100_ep3_adapter/adapter_model.safetensors"] = dict(sha256="0", bytes=1)
            json.dump(m, open(os.path.join(pod, "release", "RELEASE_SHA256.json"), "w"))
        mac = os.path.join(tmp, "mac"); dest = os.path.join(tmp, "models")
        return tmp, pod, mac, dest

    def _fetch(self, pod, mac, dest):
        return subprocess.run([sys.executable, "-I", "-B", os.path.join(SC, "e5_mac_fetch.py"), "--local-pod-root", pod, "--mac-root", mac, "--dest", dest],
                              capture_output=True, text=True)

    def test_ok(self):
        tmp, pod, mac, dest = self._pod()
        r = self._fetch(pod, mac, dest)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(dest, "WEIGHTS_OK")) and os.path.exists(os.path.join(mac, "WEIGHTS_OK")))
        self.assertTrue(os.path.exists(os.path.join(pod, "WEIGHTS_OK")))
        self.assertFalse(os.path.exists(os.path.join(pod, "DO_NOT_DELETE_POD")))
        self.assertTrue(os.path.exists(os.path.join(dest, "T100_ep3_resta0.5", "model.safetensors")))
        shutil.rmtree(tmp)

    def test_corrupt_keeps_pod(self):
        tmp, pod, mac, dest = self._pod(corrupt=True)
        r = self._fetch(pod, mac, dest)
        self.assertEqual(r.returncode, 1)
        self.assertTrue(os.path.exists(os.path.join(mac, "DO_NOT_DELETE_POD")))
        self.assertTrue(os.path.exists(os.path.join(pod, "DO_NOT_DELETE_POD")))
        self.assertFalse(os.path.exists(os.path.join(mac, "WEIGHTS_OK")))
        shutil.rmtree(tmp)

    def test_t100_adapter_refused(self):
        tmp, pod, mac, dest = self._pod(extra_t100=True)
        r = self._fetch(pod, mac, dest)
        self.assertEqual(r.returncode, 1); self.assertFalse(os.path.exists(dest + "/WEIGHTS_OK")); shutil.rmtree(tmp)

    def test_results_with_mac_only_log(self):
        tmp, pod, mac, dest = self._pod()
        for p_, b in (("eval/base/e1_judge.jsonl", b'{"id":"e1-0"}\n'), ("logs/uv_sync.log", b"pod log"), ("state/events.jsonl", b"{}\n"),
                      ("train/C100/DONE.json", b"{}")):
            os.makedirs(os.path.dirname(os.path.join(pod, p_)), exist_ok=True); open(os.path.join(pod, p_), "wb").write(b)
        os.makedirs(os.path.join(mac, "logs")); open(os.path.join(mac, "logs", "uv_sync.log"), "w").write("mac only log")
        open(os.path.join(mac, "logs", "e5_curate.log"), "w").write("mac only")
        r = subprocess.run([sys.executable, "-I", "-B", os.path.join(SC, "e5_mac_fetch.py"), "--local-pod-root", pod, "--mac-root", mac, "--dest", dest, "--results"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(open(os.path.join(mac, "logs", "uv_sync.log")).read(), "mac only log")  # 맥 로그를 덮어쓰지 않음
        self.assertEqual(open(os.path.join(mac, "logs_pod", "uv_sync.log")).read(), "pod log")
        self.assertFalse(os.path.exists(os.path.join(mac, "train", "C100", "ep3", "adapter_model.safetensors")))
        self.assertFalse(os.path.exists(os.path.join(pod, "DO_NOT_DELETE_POD")))
        shutil.rmtree(tmp)

    def test_results_failure_keeps_marker(self):
        tmp, pod, mac, dest = self._pod()
        os.makedirs(os.path.join(pod, "eval", "base")); open(os.path.join(pod, "eval", "base", "x.jsonl"), "w").write("a")
        fake = os.path.join(tmp, "bad_sha.py")  # 포드 sha256 계산을 망가뜨려 결과 대조 실패를 만든다
        r = subprocess.run([sys.executable, "-I", "-B", "-c",
                            "import sys,runpy; sys.argv=sys.argv[1:]; import importlib.util as u; "
                            "spec=u.spec_from_file_location('f', sys.argv[0]); m=u.module_from_spec(spec); spec.loader.exec_module(m); "
                            "orig=m.Pod.sha256; m.Pod.sha256=lambda self, rels: {k:('0'*64 if k.startswith('eval/') else v) for k,v in orig(self, rels).items()}; "
                            "sys.exit(m.main())",
                            os.path.join(SC, "e5_mac_fetch.py"), "--local-pod-root", pod, "--mac-root", mac, "--dest", dest, "--results"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(pod, "DO_NOT_DELETE_POD")))
        self.assertTrue(os.path.exists(os.path.join(mac, "DO_NOT_DELETE_POD")))
        shutil.rmtree(tmp)

    def test_not_ready(self):
        tmp, pod, mac, dest = self._pod(); os.remove(os.path.join(pod, "RELEASE_READY"))
        r = self._fetch(pod, mac, dest)
        self.assertEqual(r.returncode, 1); self.assertTrue(os.path.exists(os.path.join(mac, "DO_NOT_DELETE_POD"))); shutil.rmtree(tmp)


class TestAuditkitPath(unittest.TestCase):
    def test_e5_parts_on_fake_labels(self):
        tmp, root = make_root()
        try:
            env = dict(os.environ, FAKE_SEL_H="0.25:40")
            for tag, spec in (("base", {"kind": "base"}), ("T100_ep3_resta0.25", {"kind": "resta", "alpha": 0.25})):
                subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "fake_runner_e5.py"), "e5_eval.py", tag, json.dumps(spec), root], check=True, env=env)
            out = os.path.join(tmp, "ak")
            r = subprocess.run([sys.executable, "-I", "-B", os.path.join(SC, "e5_auditkit.py"), os.path.join(root, "eval"), out, "--models", "base", "T100_ep3_resta0.25",
                                "--pairs", "base:T100_ep3_resta0.25", "--parts", "e1", "mjen", "sel_h", "sel_b", "mmlu_rep", "mmlu_sel"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            res = json.load(open(os.path.join(out, "auditkit_compare.json")))
            self.assertTrue(res["all_rates_match"])
            self.assertEqual(res["rates"]["T100_ep3_resta0.25"]["sel_h"]["refusal"]["k"], 40)
            self.assertEqual(res["rates"]["T100_ep3_resta0.25"]["sel_b"]["refusal"]["k"], 5)
            self.assertEqual(res["rates"]["T100_ep3_resta0.25"]["sel_h"]["n"], 100)
            self.assertEqual(res["rates"]["base"]["mmlu_sel"]["n"], 200)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
