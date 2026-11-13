"""리포트 §5가 쓰는 툴킷 API가 설치 버전에 있는지 import·시그니처로 확인한다."""
import inspect, importlib, json, sys, warnings
warnings.filterwarnings("ignore")
rows = []
def probe(label, modname, attr=None):
    try:
        m = importlib.import_module(modname)
        obj = m
        for a in (attr.split(".") if attr else []):
            obj = getattr(obj, a)
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(no signature)"
        rows.append(dict(label=label, ok=True, where=f"{modname}{'.'+attr if attr else ''}", sig=sig[:600]))
    except Exception as e:
        rows.append(dict(label=label, ok=False, where=f"{modname}{'.'+attr if attr else ''}", err=f"{type(e).__name__}: {str(e)[:200]}"))

import safetune, auditkit, curatorkit
probe("SafeTune recover.ReStaTrainer", "safetune.recover", "ReStaTrainer")
probe("SafeTune ReStaTrainer.__init__", "safetune.recover", "ReStaTrainer.__init__")
for meth in ("train", "run", "apply", "recover", "fit"):
    if hasattr(getattr(safetune.recover, "ReStaTrainer", object), meth):
        probe(f"SafeTune ReStaTrainer.{meth}", "safetune.recover", f"ReStaTrainer.{meth}")
probe("SafeTune runner.recover.ReStaTrainer", "safetune.runner.recover", "ReStaTrainer.__init__")
probe("SafeTune runner.recover.ReStaTrainer.apply", "safetune.runner.recover", "ReStaTrainer.apply")
probe("SafeTune recover.apply_resta", "safetune.recover", "apply_resta")
probe("SafeTune runner.harden.PlainSFTTrainer", "safetune.runner.harden", "PlainSFTTrainer.__init__")
probe("SafeTune runner.harden.PlainSFTTrainer.train", "safetune.runner.harden", "PlainSFTTrainer.train")
probe("SafeTune runner.recover.LoXTrainer", "safetune.runner.recover", "LoXTrainer.__init__")
probe("SafeTune runner.recover.CThetaTrainer", "safetune.runner.recover", "CThetaTrainer.__init__")
probe("AuditKIT ChrF", "auditkit", "ChrF.__init__")
probe("AuditKIT RunComparison.significance", "auditkit.comparison", "RunComparison.significance")
probe("AuditKIT GuardJudge", "auditkit", "GuardJudge.__init__")
probe("CuratorKIT CuratorConfig", "curatorkit", "CuratorConfig")
probe("CuratorKIT MinHashDeduplicator", "curatorkit.normalizers.dedup", "MinHashDeduplicator.__init__")
probe("SafeTune recover.task_arithmetic", "safetune.recover", "task_arithmetic")
probe("SafeTune evaluate.evaluate", "safetune.evaluate", "evaluate")
probe("AuditKIT ak.evaluate", "auditkit", "evaluate")
probe("AuditKIT ak.compare", "auditkit", "compare")
probe("CuratorKIT Curator", "curatorkit", "Curator")
json.dump(dict(versions=dict(safetune=safetune.__version__, auditkit=getattr(auditkit,"__version__",None), curatorkit=getattr(curatorkit,"__version__",None)),
               recover_all=sorted(n for n in dir(safetune.recover) if not n.startswith("_")),
               auditkit_all=sorted(n for n in dir(auditkit) if not n.startswith("_")),
               curatorkit_all=sorted(n for n in dir(curatorkit) if not n.startswith("_")),
               rows=rows), sys.stdout, ensure_ascii=False, indent=1)
