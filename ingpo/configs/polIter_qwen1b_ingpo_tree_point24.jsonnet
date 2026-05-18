// InGPO-tree on Point24 (24-game style) with DeepSeek-R1-Distill-Qwen-1.5B.
// Mirrors SPO's polIter_qwen1b_spo_tree_point24.jsonnet so we can show
// InGPO results on a non-MATH reasoning benchmark.
(import 'polIter_qwen1b_spo_tree_point24.jsonnet')
+ (import 'ingpo_defaults.libsonnet')
+ (import 'ingpo_overlay.libsonnet')
