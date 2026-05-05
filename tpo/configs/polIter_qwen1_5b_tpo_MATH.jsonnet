// TPO training config for Qwen2.5-1.5B on MATH dataset.
//
// Mirrors SPO's polIter_qwen1_5b_base_spo_chain_MATH.jsonnet with:
//   - inference_strategy: "tpo"  (entropy branching + P-degradation termination)
//   - episode_generator:  "tpo"  (V=0 for early-stopped leaves)
//   - All PPO hyperparameters identical to SPO.
//
// Usage:
//   cd /home/user/adpo/spo
//   APP_SEED=42 python -m treetune.main \
//       ../tpo/configs/polIter_qwen1_5b_tpo_MATH.jsonnet
//
// Environment variables (override via shell or APP_ prefix):
//   TPO_SERVER_URL   vLLM server URL (default: http://localhost:8000/v1)
//   TPO_MODEL_NAME   Model identifier on the vLLM server
//   APP_SEED         Random seed

local hf_model_name = 'Qwen/Qwen2.5-1.5B';

local math_task = (import '../../../spo/configs/tasks/math_inplace_no_answer_prefix.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

// Training volume
local num_episodes_per_iteration = 512;
local num_rollouts_per_sample    = 8;     // TPO trees per question
local num_dataset_samples        = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations       = 1000;

local sampling_temperature = 0.6;

// TPO hyperparameters
local tpo_min_branches       = 2;
local tpo_max_branches       = 8;
local tpo_max_degradations   = 3;   // K: terminate after K P-degradations
local tpo_max_depth          = 8;   // hard depth cap
local tpo_max_tokens         = 512; // tokens per reasoning step
local tpo_top_k_entropy      = 20;  // tokens for entropy estimation

// vLLM server (set via env vars or override in a +{} block)
local server_url  = std.extVar('TPO_SERVER_URL');   // e.g. "http://localhost:8000/v1"
local model_name  = std.extVar('TPO_MODEL_NAME');   // e.g. "Qwen/Qwen2.5-1.5B"

(import '../../../spo/configs/gvar.jsonnet')
+ (import '../../../spo/configs/runtimes/policy_iteration.jsonnet')
+ (import '../../../spo/configs/trainers/ppo_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'tpo',     // registered by tpo.training module import

    // Pass the math task for answer grading
    task: math_task,

    initial_model_name_or_path: hf_model_name,

    dataset_num_samples_per_iteration: num_dataset_samples,
    total_num_iterations: $.num_iterations,

    vllm_gpu_memory_utilization: 0.4,
    vllm_min_available_gpu_memory_mb: 4 * 1024,
    wait_until_memory_release: true,
    vllm_server+: {
        swap_space: 32,
        max_num_seqs: 512,
        enable_prefix_caching: true,
    },

    max_sequence_length: null,
    max_question_length: 512,

    append_bos_to_query: false,
    append_eos_to_response: false,

    dataset_shuffle_on_each_iteration: true,
    dataset_shuffle_before_portion: true,
    dataset_sample_with_replacement: true,
    fill_missing_episodes: true,

    question_template: '[MATH_TASK] Problem:\n{query}\n\nSolution:',

    // TPO inference strategy — replaces SPO's guidance-based tree builder
    inference_strategy: {
        type: 'tpo',

        server_url: server_url,
        model_name: model_name,

        question_template: '[MATH_TASK] Problem:\n{query}\n\nSolution:',
        question_field: 'query',
        answer_field: 'answer',

        // Entropy-based branching
        branching_strategy: {
            type: 'entropy',
            min_branches: tpo_min_branches,
            max_branches: tpo_max_branches,
            top_k: tpo_top_k_entropy,
        },

        // P-degradation termination
        termination_strategy: {
            type: 'p_degradation',
            max_degradations: tpo_max_degradations,
            max_depth: tpo_max_depth,
        },

        // Fixed-token segmentation (same as SPO's M-token mode)
        segmentation_strategy: {
            type: 'fixed_token',
            max_tokens: tpo_max_tokens,
            temperature: sampling_temperature,
            top_p: 1.0,
        },

        get_logprobs: true,
        top_k_entropy: tpo_top_k_entropy,
        max_workers: 4,
    },

    reward_function: {
        type: 'math_reward_function',
        penalize_unfinished_response: true,
        unfinished_response_penalty: 0.0,
        math_task: $.episode_generator.task,
    },
  },

  tokenizer: {
    type: 'pretrained',
    hf_model_name: $.episode_generator.initial_model_name_or_path,
  },
  use_deepspeed: true,

  num_iterations: total_num_iterations,
  num_episodes_per_iteration: num_episodes_per_iteration,
  episodes_cloud_log_steps: 50,

  trainer+: {
    params+: {
        temperature: sampling_temperature,
        use_prob_mask: true,
    },

    num_epochs_per_iteration: 1,

    actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

    critic_model: null,
    critic_deepspeed_config: null,
    save_hf_critic_checkpoint: false,

    reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

    actor_deepspeed_config: (import '../../../spo/configs/deepspeed/zero_0.jsonnet'),
    move_reference_model_to_cpu: true,

    report_entropy: false,

    general_training_args+: {
        target_train_batch_size: 128,
        per_device_train_batch_size: 2,
        per_device_eval_batch_size: 2,
        gradient_accumulation_steps: null,
        save_steps: 5,
        checkpoint_keep_steps: 10,
    },
  },
}
+ (import '../../../spo/configs/trainers/lam1.jsonnet')
+ (import '../../../spo/configs/trainers/refKl0.0001.jsonnet')
+ (import '../../../spo/configs/trainers/klLoss.jsonnet')
