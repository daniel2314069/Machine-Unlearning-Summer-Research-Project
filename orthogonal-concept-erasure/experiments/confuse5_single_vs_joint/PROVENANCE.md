# Resolved protocol provenance

Primary label: **official released repository behavior with explicitly resolved
paper and benchmark supplements**. It must not be shortened to
“paper-aligned.” Runtime source hashes, Git state, package versions, model
component identities/commit hashes, K0 identity, and checkpoint hashes are
written into the generated metadata.

Precedence is:

1. official task experiment script or execution path;
2. final-paper Appendix/main-text setting when the task path is silent;
3. README or generic config;
4. parser defaults are forbidden.

## Explicit resolutions

| Detail | Official released path | Final-paper statement | Primary resolution | Category |
|---|---|---|---|---|
| Object coefficients | `trainscripts/object.sh`: 2000 / 10 / 0 | Appendix C: 1000 / 50 / 1 | 1000 / 50 / 1, identically for Single and Joint | User-resolved paper supplement; conflict recorded |
| Extra `lamb` term | `object.sh`: 10; `oce.py` applies `lamb*(W0@W0.T)` | Absent from Appendix C Eq. 35 | 10 | Official-repo behavior |
| Reflection handling | `oce.py` flips the final column of computed `R` when raw `det(R)<0` | Closed form gives `UV^T`; no correction stated | Correction on | Official-repo behavior |
| Object prompt expansion | `object.sh` passes true; `oce.py` retains all bare concepts, then appends five extras per concept | Object protocol uses expansion | On, with exact released ordering | Official-repo behavior |
| Target input | `oce.py` L2-normalizes each projected concept vector, epsilon `1e-8`, then reduced QR | Subspace construction | Exact released behavior | Official-repo behavior |
| Erasure order | `oce.py`: `-R(I-R*)` | Appendix C agrees; another paper presentation differs | `-R(I-R*)` | Task path and Appendix C |
| Global prior | `oce.py` consumes `Cg.pt`; upstream computation is nonpadding CLIP token second moment | Appendix defines the global second moment | Fresh audited all-row COCO-30k K0, no resume | Paper definition with cleaned computation path |
| Local retain `C_n` | Task script does not define Confuse5 neighbors | General local-retain term | The three designated similar non-targets; anchors excluded | Benchmark-specific choice |
| Semantic anchors | Task example supplies a related guide; paper gives heuristic guidance | Same high-level category, related but noticeably different | Fixed `anchors.json`; identical in Single and Joint | Paper heuristic plus benchmark mapping |
| Generation/evaluator | Not part of OCE checkpoint objective | Not a Confuse5 matched protocol | Existing 12,500 rows, PNDM/50/7.5/512, bfloat16, pinned ResNet-50 V2 | Benchmark-specific choice |

The checkpoint diagnostic evaluates the same resolved target, anchor, retain,
K0, and layers twice without generating a second formal corpus:

- primary: `lamb=10`, released determinant correction on;
- paper-literal diagnostic: `lamb=0`, determinant correction off.

For every layer it records raw and final determinants, whether correction was
triggered, `||P-I||F`, `||PW-W||F`, target/anchor subspace displacement,
objective component traces, retain error, and `||P_repo-P_paper||F`.

## Benchmark gate operationalization

- Anchor collision failure is at least 4 exact target top-1 labels among the
  target-associated ordered 8 seeds. This operationalizes “large collision”
  without using edit outcomes and is fixed before checkpoint construction.
- The Original canary requires all 128 PNG hashes to match the archived
  manifests exactly.
- Each of the four Single smoke targets must reduce exact target top-1 by at
  least 4/32. Joint results are reported but never gate the formal run.

The archived Original is eligible only after the canary and complete
prompt/seed/generation/evaluator validation. Its full-run auxiliary metrics are
unavailable by design.
