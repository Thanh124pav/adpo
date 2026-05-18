// Helper to import SPO configs from the cloned SPO repo.
// All InGPO configs do `(import '_paths.libsonnet').spo('configs/foo.jsonnet')`.

{
  // Path to SPO configs relative to ingpo/ (the directory `configs/` lives in).
  spo_configs:: '',
  spo(rel):: import './' + rel,  // jsonnet cannot resolve at lib time; users
  // should set jsonpath include path to include both ingpo/configs/ and
  // ingpo/spo/configs/ when invoking jsonnet. This file documents the conv.
}
