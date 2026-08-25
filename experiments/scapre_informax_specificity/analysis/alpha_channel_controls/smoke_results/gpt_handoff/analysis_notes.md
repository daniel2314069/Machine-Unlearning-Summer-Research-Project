# Alpha-channel controls smoke: retrieval and analysis notes

## Retrieval verification

- Archive: `scapre_informax_alpha_channel_controls_smoke_20260825T185142Z_20260825T190124Z.tar.gz`
- Archive SHA-256: `83c5cabdac7a1bfb546538e6b75eff38bd3d5bfc6f79ba61148db7ba7bcb67ab`
- Cleanup sidecar status: `passed`
- Archive path audit: 91 entries, all safe relative paths
- Server commit: `2f524dc8bc4527f2246be75a5f353b20fb2e0556` on clean `main`
- Server environment: Conda `MU`
- Generated images: 60; all were hashed before cleanup
- Image-manifest SHA-256: `893e7ed709c29fb9fdabf549bf982a0f7fc596e1cbc3adfdd86ae7c0ee366feb`
- Cleanup: 60 PNGs removed only after archive verification

The full extracted archive, including `.pt` diagnostics and logs, is retained
outside Git under:

```text
.local_artifacts/scapre_informax/scapre_informax_alpha_channel_controls_smoke_20260825T185142Z_20260825T190124Z/
```

The CSV copies committed under `smoke_results/` use LF line endings for Git and
web-tool compatibility. The server originals use CRLF and remain byte-for-byte
unchanged in `.local_artifacts`; consequently, the archived `result_manifest.json`
hashes refer to those originals rather than the normalized Git copies.

## Integrity outcome

All smoke gates passed:

- official empty-string neutral only;
- identical raw MI, pre-aggregate alpha, thresholds, and non-alpha edit inputs;
- identical prompt lists, generation seeds, and evaluator fingerprint;
- per-matrix constant means within tolerance (maximum absolute error
  `1.3358658172701876e-09`);
- exact alpha multiset preservation for all 96 shuffled matrix records (three
  shuffle salts x 32 matrices);
- exact all-one alpha for all 32 `identity_B` matrices;
- all edited projection weights finite;
- six distinct checkpoint hashes, confirming distinct interventions.

## Smoke image observations

The smoke denominator is only one edit seed, one Confuse5 group, five concepts,
and two images per concept: 10 images per variant.

All six variants have identical coarse metrics:

| Variant | Unlearn | Preserve | Overall |
| --- | ---: | ---: | ---: |
| official | 0.00 | 50.00 | 66.67 |
| constant_mean | 0.00 | 50.00 | 66.67 |
| shuffled | 0.00 | 50.00 | 66.67 |
| shuffled_alt1 | 0.00 | 50.00 | 66.67 |
| shuffled_alt2 | 0.00 | 50.00 | 66.67 |
| identity_B | 0.00 | 50.00 | 66.67 |

Equal accuracy does not mean equal outputs:

| Variant | Classifier-label differences vs official | Correctness differences | Byte-identical images |
| --- | ---: | ---: | ---: |
| constant_mean | 1/10 | 0/10 | 0/10 |
| shuffled | 0/10 | 0/10 | 0/10 |
| shuffled_alt1 | 1/10 | 0/10 | 0/10 |
| shuffled_alt2 | 1/10 | 0/10 | 0/10 |
| identity_B | 3/10 | 0/10 | 0/10 |

The official per-matrix mean alpha ranges from approximately `0.00603` to
`0.00933`; `identity_B` sets it to `1`, roughly 107-166 times those means.
Its larger prediction-label movement is therefore consistent with a materially
different scale, but 10 images cannot establish an image-level advantage or
cost.

## Decision boundary

This smoke supports only the claim that the implementation, checkpoint,
generation, evaluator, shuffle, and cleanup paths work and that the controls
produce genuinely different model/image outputs. It provides no scientific
evidence that channel identity is or is not important because all reported
accuracies have very small denominators and are tied.

Formal remains the experiment that can answer the registered question: five
edit seeds, all 25 concepts, 3,000 score rows per variant/seed, and the already
verified official score reuse.
