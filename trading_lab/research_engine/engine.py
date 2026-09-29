from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ResearchBudget:
    max_experiments_per_cycle: int = 20
    max_experiments_total: int = 500
    max_generations: int = 3
    stop_after_no_improvement: int = 5
    max_runtime_minutes: int = 120

    def __post_init__(self):
        if min(self.max_experiments_per_cycle, self.max_experiments_total,
               self.max_generations, self.stop_after_no_improvement,
               self.max_runtime_minutes) <= 0:
            raise ValueError("Research budget values must be positive.")


def run_bounded_cycle(registry, dataset_hash, code_version, hypotheses, evaluator,
                      budget=None, improvement_test=None):
    """Run a finite iterable of hypotheses.

    evaluator(hypothesis) -> (metrics, robustness). The function never creates new
    hypotheses itself and therefore cannot silently expand the search space.
    """
    budget = budget or ResearchBudget()
    improvement_test = improvement_test or (lambda metrics, robustness: bool(robustness.get("passes", False)))
    started = time.monotonic()
    results, no_improvement = [], 0
    for idx, hyp in enumerate(hypotheses):
        if idx >= budget.max_experiments_per_cycle:
            break
        if registry.count() >= budget.max_experiments_total:
            break
        if time.monotonic() - started >= budget.max_runtime_minutes * 60:
            break
        params = dict(hyp.get("params", {}))
        model = str(hyp.get("model", "UNKNOWN"))
        hypothesis_id = str(hyp.get("hypothesis_id", f"H{idx+1:04d}"))
        exp_id, created = registry.start(hypothesis_id, dataset_hash, code_version, model, params)
        if not created:
            results.append({"experiment_id": exp_id, "status": "DUPLICATE"})
            continue
        t0 = time.monotonic()
        try:
            metrics, robustness = evaluator(hyp)
            passed = bool(improvement_test(metrics, robustness))
            decision = "CANDIDATE" if passed else "REJECTED"
            reason = robustness.get("reason", "robustness criteria")
            registry.finish(exp_id, metrics=metrics, robustness=robustness, decision=decision,
                            reason=reason, runtime_seconds=time.monotonic()-t0)
            results.append({"experiment_id": exp_id, "status": "COMPLETED", "decision": decision,
                            "metrics": metrics, "robustness": robustness})
            if passed:
                no_improvement = 0
            else:
                no_improvement += 1
        except Exception as exc:
            registry.finish(exp_id, status="FAILED", decision="REJECTED", reason=str(exc),
                            runtime_seconds=time.monotonic()-t0)
            results.append({"experiment_id": exp_id, "status": "FAILED", "reason": str(exc)})
            no_improvement += 1
        if no_improvement >= budget.stop_after_no_improvement:
            break
    stop_reason = "budget_or_input_exhausted"
    if no_improvement >= budget.stop_after_no_improvement:
        stop_reason = "no_robust_improvement_limit"
    elif registry.count() >= budget.max_experiments_total:
        stop_reason = "total_experiment_budget"
    elif time.monotonic() - started >= budget.max_runtime_minutes * 60:
        stop_reason = "runtime_budget"
    return {"results": results, "stop_reason": stop_reason,
            "experiments_this_cycle": len(results), "experiments_total": registry.count()}
