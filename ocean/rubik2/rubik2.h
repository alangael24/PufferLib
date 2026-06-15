#pragma once

#define RUBIK_MAX_FIXED_MIX 16
#define c_reset rubik3_c_reset
#define c_step rubik3_c_step
#define c_render rubik3_c_render
#define c_close rubik3_c_close
#include "../rubik/rubik.h"
#undef c_reset
#undef c_step
#undef c_render
#undef c_close

static inline int rubik2_is_corner_idx(int idx) {
    int pos = idx % RUBIK_FACE_STICKERS;
    return pos == 0 || pos == 2 || pos == 6 || pos == 8;
}

static inline int rubik2_distance(Rubik* env) {
    int wrong = 0;
    for (int face = 0; face < RUBIK_FACES; face++) {
        for (int i = 0; i < RUBIK_FACE_STICKERS; i++) {
            int idx = face * RUBIK_FACE_STICKERS + i;
            if (!rubik2_is_corner_idx(idx)) {
                continue;
            }
            wrong += env->state.cube[idx] != (unsigned char)face;
        }
    }
    return wrong;
}

static inline void rubik2_turn(Rubik* env, int action) {
    int axis;
    int layer;
    int turns;
    unsigned char next[RUBIK_STICKERS];
    memcpy(next, env->state.cube, RUBIK_STICKERS * sizeof(unsigned char));
    rubik_action_to_layer(action, &axis, &layer, &turns);

    for (int idx = 0; idx < RUBIK_STICKERS; idx++) {
        if (!rubik2_is_corner_idx(idx)) {
            continue;
        }

        int x;
        int y;
        int z;
        int nx;
        int ny;
        int nz;
        rubik_sticker_to_coord(idx, &x, &y, &z, &nx, &ny, &nz);

        int coord = axis == 0 ? x : (axis == 1 ? y : z);
        if (coord != layer) {
            continue;
        }

        rubik_rotate_vec(axis, turns, &x, &y, &z);
        rubik_rotate_vec(axis, turns, &nx, &ny, &nz);
        int dst = rubik_coord_to_sticker(x, y, z, nx, ny, nz);
        next[dst] = env->state.cube[idx];
    }

    memcpy(env->state.cube, next, RUBIK_STICKERS * sizeof(unsigned char));
}

static inline void rubik2_scramble(Rubik* env) {
    rubik_set_solved(env);
    rubik_select_episode_depth(env);
    int depth = rubik_clamp(env->scramble_depth, 0, 1000);
    int last_face = -1;

    for (int i = 0; i < depth; i++) {
        int face = rand_r(&env->rng) % RUBIK_FACES;
        if (depth > 1) {
            while (face == last_face) {
                face = rand_r(&env->rng) % RUBIK_FACES;
            }
        }
        int turn = rand_r(&env->rng) % 3;
        rubik2_turn(env, face * 3 + turn);
        last_face = face;
    }
}

static inline void rubik2_start_episode(Rubik* env) {
    env->state.step_count = 0;
    env->state.episode_return = 0.0f;
    env->state.last_action = -1;
    rubik2_scramble(env);
    env->state.distance = rubik2_distance(env);
    env->state.best_distance = env->state.distance;
    rubik_compute_observations(env);
}

void c_reset(Rubik* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    rubik2_start_episode(env);
}

void c_step(Rubik* env) {
    int action = (int)env->actions[0];
    env->terminals[0] = 0.0f;

    int prev_distance = env->state.distance;
    float reward = -env->step_penalty;

    if (action < 0 || action >= RUBIK_ACTIONS) {
        reward -= 0.25f;
    } else {
        rubik2_turn(env, action);
        if (action == env->state.last_action) {
            reward -= 0.0025f;
        }
        env->state.last_action = action;
    }

    env->state.step_count += 1;
    env->state.distance = rubik2_distance(env);
    reward += env->progress_reward * (float)(prev_distance - env->state.distance);

    if (env->state.distance < env->state.best_distance) {
        env->state.best_distance = env->state.distance;
        reward += 0.05f;
    }

    int solved = env->state.distance == 0;
    if (solved) {
        reward += env->solve_reward;
        env->terminals[0] = 1.0f;
    } else if (env->state.step_count >= env->max_steps) {
        reward -= env->timeout_penalty;
        env->terminals[0] = 1.0f;
    }

    env->rewards[0] = reward;
    env->state.episode_return += reward;

    if (env->terminals[0] > 0.0f) {
        rubik_add_log(env, solved);
        rubik_update_curriculum(env, solved);
        if (env->auto_reset_enabled) {
            rubik2_start_episode(env);
        } else {
            rubik_compute_observations(env);
        }
    } else {
        rubik_compute_observations(env);
    }
}

void c_render(Rubik* env) {
    rubik3_c_render(env);
}

void c_close(Rubik* env) {
    rubik3_c_close(env);
}
