// InGPO-tree on MATH with Rho-1.1B-SFT.
// Useful for cross-dataset ablations (compare to *_GSM8K twin).
(import '../spo/configs/polIter_rho1bSft2_spo_tree_MATH.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ (import 'ingpo_overlay.libsonnet')
