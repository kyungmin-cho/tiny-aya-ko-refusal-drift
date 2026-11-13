"""E4 실제 실행 시간 집계(맥, 표준 라이브러리만): state/events.jsonl과 train/<arm>/DONE.json, eval/<tag>/DONE.json에서 측정값만 모은다.
출력: agg/e4_timing.json. 요금 $0.59/h는 과제 지시문의 포드 요금(측정 아님)으로 곱한 추정 비용이다.
사용: python3 -I experiments/e4/scripts/e4_timing.py experiments/e4
"""
import datetime, json, os, sys

RATE = 0.59


def ts(s):
    return datetime.datetime.fromisoformat(s)


def main(root):
    ev = [json.loads(l) for l in open(os.path.join(root, "state", "events.jsonl"), encoding="utf-8") if l.strip()]
    starts = {e["step"]: e for e in ev if e["event"] == "start"}
    ends = {e["step"]: e for e in ev if e["event"] == "end"}
    steps = {}
    for k, e in ends.items():
        steps[k] = dict(start=starts.get(k, {}).get("kst"), end=e["kst"], sec=e["sec"], ok=e["ok"], est_min=starts.get(k, {}).get("est_min"))
    train = {}
    for arm in ("T100", "C100", "T200", "C200"):
        p = os.path.join(root, "train", arm, "DONE.json")
        if os.path.exists(p):
            d = json.load(open(p))
            train[arm] = {k: d.get(k) for k in ("global_steps", "train_sec", "sec_per_step", "load_sec", "peak_mem_allocated_gb", "first_loss", "last_loss", "nan_losses")}
    evals = {}
    for tag in sorted(os.listdir(os.path.join(root, "eval"))) if os.path.isdir(os.path.join(root, "eval")) else []:
        p = os.path.join(root, "eval", tag, "model_info.json")
        if os.path.exists(p):
            d = json.load(open(p))
            evals[tag] = {k: d.get(k) for k in ("model_build_sec", "gen_wall_sec", "judge_wall_sec")}
            if "resta" in d:
                evals[tag]["resta_sec"] = d["resta"]["sec"]
    t_first = min(ts(e["kst"]) for e in ev); t_last = max(ts(e["kst"]) for e in ev)
    busy = sum(s["sec"] for k, s in steps.items() if k not in ("preflight",))
    wall = (t_last - t_first).total_seconds()
    out = dict(orchestrator_first_event=t_first.isoformat(), orchestrator_last_event=t_last.isoformat(),
               orchestrator_wall_min=round(wall / 60, 1), step_sum_min=round(busy / 60, 1), steps=steps, train=train, eval=evals,
               rate_usd_per_h=RATE, note="step_sum_min = 학습·평가·집계 하위 프로세스 시간 합(GPU를 잡고 있던 시간의 측정값). 비용 = 시간 × 지시문 요금(추정)")
    json.dump(out, open(os.path.join(root, "agg", "e4_timing.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("steps", "eval")}, ensure_ascii=False, indent=1))
    for k, s in steps.items():
        print(f"{k:<28} {s['sec']/60:6.1f}분 (예상 {s['est_min']}) ok={s['ok']}")


if __name__ == "__main__":
    main(sys.argv[1])


def recompute_after_first_train(root, t_ref="2026-11-12T10:10:21+09:00", sec_per_step=1.3019):
    """T100 학습이 끝난 시점(t_ref)에서, S1 오케스트레이터 시간 가정 중 스텝 시간만 T100 실측(DONE.json sec_per_step)으로 바꿔
    남은 필수 단계 시간을 다시 더한다. 다른 가정(적재 5.4초, 학습 부가 20초, 평가 266.6초, ReSta +180초, 집계 5초)은 S1 값 그대로."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
    import e4_orchestrate as O
    T = dict(O.T, sec_per_step=sec_per_step); O.T.update(T)
    rest = [s for s in O.plan() if not s["optional"] and s["name"] not in ("preflight", "train_T100")]
    sec = sum(O.est_sec(s) for s in rest)
    end = ts(t_ref) + datetime.timedelta(seconds=sec)
    return dict(t_ref=t_ref, sec_per_step_measured=sec_per_step, remaining_required_min=round(sec / 60, 1), predicted_end=end.isoformat(timespec="seconds"),
                steps=[s["name"] for s in rest], note="S1 가정 중 스텝 시간만 실측으로 바꾼 재계산(추정). 실제 필수 단계 종료는 state/events.jsonl의 eval_T200_ep3_resta0.25 end")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--recompute":
    r = recompute_after_first_train(sys.argv[1])
    p = os.path.join(sys.argv[1], "agg", "e4_timing.json"); d = json.load(open(p)); d["recompute_after_T100"] = r
    json.dump(d, open(p, "w"), ensure_ascii=False, indent=1); print(r)
