# OCE zero-generation qualification audit

1. **Direction 1: GO.** Off-diagonal update-energy fraction across 64 target/layer rows: median 0.831538, p95 0.861836, range [0.676542, 0.863279]. The fraction of rows at or above 0.10 is 1.000. Runtime head layouts: 1280x768: 8 heads x 160, 320x768: 8 heads x 40, 640x768: 8 heads x 80. Target medians: airplane=0.832834, bird=0.829476, dog=0.833907, truck=0.831027.

2. **Direction 3: GO.** The official determinant correction triggered in 12/64 rows (0.188); per target: airplane=3/16, bird=4/16, dog=1/16, truck=4/16. M rank/nullity: 1280x1280: rank 767-768, nullity 512-513, 320x320: rank 302-320, nullity 0-18, 640x640: rank 563-625, nullity 15-77. 12 triggered rows were numerically rank-deficient. Maximum correction-induced edited-weight relative difference was 6.657136e-02. Across 2 CPU float64 case(s), the maximum difference between two legal numerical-null-space SVD realizations after the official correction was 5.529539e-02.

3. **Next minimum image-level qualification (not run):**

   - Direction 1: use one representative low/mid/high-resolution layer group and a tiny fixed-seed target-prompt set to compare the official edit with a head-local matched control.
   - Direction 3: use the smallest triggered rank-deficient case and a tiny fixed-seed target/preservation prompt set to compare weights from two recorded legal SVD realizations.

No images were generated. Full per-layer evidence is in `direction1_layers.csv`, `direction3_layers.csv`, and `audit_results.json`.
