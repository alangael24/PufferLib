#include <stdio.h>
#include "rubik2.h"

#define OBS_SIZE RUBIK_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {RUBIK_ACTIONS}
#define OBS_TENSOR_T ByteTensor

#define Env Rubik
static inline void puffer_state_refresh(Rubik* env) { rubik_compute_observations(env); }
#include "vecenv.h"

static inline float safe_ratio(float numerator, float denominator) {
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    return numerator / denominator;
}

static inline int dict_get_int_default(Dict* kwargs, const char* key, int default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return default_value;
    }
    return (int)item->value;
}

static inline float dict_get_float_default(Dict* kwargs, const char* key, float default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item == NULL) {
        return default_value;
    }
    return (float)item->value;
}

static inline void dict_set_depth_log(Log* log, Dict* out, int depth) {
    char key[48];
    snprintf(key, sizeof(key), "solve_rate_depth_%d", depth);
    dict_set(out, key, safe_ratio(log->depth_solved[depth], log->depth_n[depth]));
    snprintf(key, sizeof(key), "distance_depth_%d", depth);
    dict_set(out, key, safe_ratio(log->depth_distance[depth], log->depth_n[depth]));
    snprintf(key, sizeof(key), "episode_return_depth_%d", depth);
    dict_set(out, key, safe_ratio(log->depth_return[depth], log->depth_n[depth]));
    snprintf(key, sizeof(key), "frac_depth_%d", depth);
    dict_set(out, key, log->depth_n[depth]);
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->scramble_depth = (int)dict_get(kwargs, "scramble_depth")->value;
    env->curriculum_enabled = (int)dict_get(kwargs, "curriculum_enabled")->value;
    env->curriculum_depth_0 = (int)dict_get(kwargs, "curriculum_depth_0")->value;
    env->curriculum_depth_1 = (int)dict_get(kwargs, "curriculum_depth_1")->value;
    env->curriculum_depth_2 = (int)dict_get(kwargs, "curriculum_depth_2")->value;
    env->curriculum_depth_3 = (int)dict_get(kwargs, "curriculum_depth_3")->value;
    env->fixed_mixed_enabled = (int)dict_get(kwargs, "fixed_mixed_enabled")->value;
    env->auto_reset_enabled = dict_get_int_default(kwargs, "auto_reset_enabled", 1);
    env->fixed_depth_8_prob = (float)dict_get(kwargs, "fixed_depth_8_prob")->value;
    env->fixed_depth_9_prob = (float)dict_get(kwargs, "fixed_depth_9_prob")->value;
    env->fixed_depth_10_prob = (float)dict_get(kwargs, "fixed_depth_10_prob")->value;
    env->fixed_mixed_count = dict_get_int_default(kwargs, "fixed_mixed_count", 0);
    if (env->fixed_mixed_count < 0) {
        env->fixed_mixed_count = 0;
    }
    if (env->fixed_mixed_count > RUBIK_MAX_FIXED_MIX) {
        env->fixed_mixed_count = RUBIK_MAX_FIXED_MIX;
    }
    for (int i = 0; i < RUBIK_MAX_FIXED_MIX; i++) {
        char depth_key[32];
        char prob_key[32];
        snprintf(depth_key, sizeof(depth_key), "fixed_depth_%d", i);
        snprintf(prob_key, sizeof(prob_key), "fixed_prob_%d", i);
        env->fixed_mixed_depths[i] = dict_get_int_default(kwargs, depth_key, -1);
        env->fixed_mixed_probs[i] = dict_get_float_default(kwargs, prob_key, 0.0f);
    }
    env->curriculum_min_episodes = (int)dict_get(kwargs, "curriculum_min_episodes")->value;
    env->curriculum_success_threshold =
        (float)dict_get(kwargs, "curriculum_success_threshold")->value;
    env->curriculum_ema_alpha = (float)dict_get(kwargs, "curriculum_ema_alpha")->value;
    env->step_penalty = (float)dict_get(kwargs, "step_penalty")->value;
    env->progress_reward = (float)dict_get(kwargs, "progress_reward")->value;
    env->solve_reward = (float)dict_get(kwargs, "solve_reward")->value;
    env->timeout_penalty = (float)dict_get(kwargs, "timeout_penalty")->value;
    env->curriculum_stage = 0;
    env->curriculum_stage_episodes = 0;
    env->curriculum_success_ema = 0.0f;
    rubik_refresh_curriculum_depth(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "solved", log->solved);
    dict_set(out, "distance", log->distance);
    dict_set(out, "scramble_depth", log->scramble_depth);
    dict_set(out, "curriculum_stage", log->curriculum_stage);
    dict_set(out, "solve_rate_ema", log->solve_rate_ema);
    for (int depth = 1; depth < RUBIK_LOG_DEPTHS; depth++) {
        if (log->depth_n[depth] > 0.0f) {
            dict_set_depth_log(log, out, depth);
        }
    }
    dict_set(out, "avg_return_solved_episodes",
        safe_ratio(log->solved_return, log->solved_return_n));
    dict_set(out, "avg_return_unsolved_episodes",
        safe_ratio(log->unsolved_return, log->unsolved_return_n));
    dict_set(out, "frac_solved_episodes", log->solved_return_n);
    dict_set(out, "frac_unsolved_episodes", log->unsolved_return_n);
}
