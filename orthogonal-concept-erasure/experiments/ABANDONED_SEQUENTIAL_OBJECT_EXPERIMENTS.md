# Abandoned: sequential OCE object experiments

**Research status: ABANDONED — not valid for paper claims or headline conclusions.**

This status applies to both:

- `sequential_object_persistence`
- `sequential_object_followup`

The runs completed and their raw evaluator records remain useful as an audit
trail. The experiment family is abandoned because the design does not identify
the causal questions it was intended to answer. Do not cite the existing tables,
figures, or generated summaries as evidence that Retain Once persists, that
Retain Always is unnecessary, or that earlier edits generally reduce the
effectiveness of later OCE requests.

## Blocking design problems

1. **The retain-persistence comparison has no Retain Never control.** Elephant
   scored 1.0 in the original model and at every checkpoint in both tested
   conditions. Without a chain that never explicitly retains elephant, the
   result cannot distinguish persistent protection from a concept that would
   have remained unchanged without protection.
2. **The elephant assay is saturated.** Both top-1 accuracy and the recorded
   expected-class probability stay effectively at the ceiling, so the assay has
   no useful sensitivity for the intended comparison.
3. **The direct-single and sequential own-step conditions are not matched.** The
   original sequential chains include elephant in the local retain request at
   W01, while direct-single checkpoints do not. Retain Always also continues to
   include elephant at every step. Even airplane at W01, where there are no
   previous edits, differs from direct-single (0.040 versus 0.015). Therefore the
   observed differences cannot be attributed solely to previous edits.
4. **Target identity, step number, and edit history are confounded.** Every
   target occupies only one position in one fixed order. The design cannot
   separate accumulation from the identities and ordering of earlier edits.
5. **The clean five-step chain is not a matched counterfactual for the original
   dog/bird trajectories.** It removes the specific event “the anchor is later
   erased,” but also changes order, step, retain history, and later edits. Bird
   and airplane show descriptive CLIP-accuracy increases in that chain, while
   dog does not reproduce, but those observations do not isolate a single cause.
6. **The evaluator is narrow.** Forced-choice CIFAR-10 CLIP target accuracy is an
   operational signal, not a complete measure of human-visible concept
   generation. Semantically close target/anchor pairs, especially automobile
   and truck, are particularly difficult to interpret as clean erasure outcomes.

## What remains valid

- The saved manifests, seeds, per-image predictions, aggregates, and image-count
  audits describe the executions that occurred.
- In the exact tested sequences, some previously erased targets later received
  higher CIFAR-10 CLIP target accuracy. Bird and airplane do so in the clean
  five-step chain; dog does not reproduce there.
- These are target- and order-specific descriptive observations only. They do
  not establish a universal sequential OCE persistence failure or its cause.

## Archival policy

All code and raw outputs are retained for reproducibility. The launchers are
also retained, but this experiment family should not receive additional runs or
be extended in place. Any replacement study must use a newly named experiment
with matched controls and a non-saturated retention measurement.
