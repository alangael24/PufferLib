#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define RUBIK_FACES 6
#define RUBIK_FACE_STICKERS 9
#define RUBIK_STICKERS (RUBIK_FACES * RUBIK_FACE_STICKERS)
#define RUBIK_OBS_SIZE (RUBIK_STICKERS * RUBIK_FACES)
#define RUBIK_ACTIONS 18
#ifndef RUBIK_MAX_FIXED_MIX
#define RUBIK_MAX_FIXED_MIX 8
#endif
#ifndef RUBIK_LOG_DEPTHS
#define RUBIK_LOG_DEPTHS 21
#endif

enum {
    RUBIK_U = 0,
    RUBIK_R = 1,
    RUBIK_F = 2,
    RUBIK_D = 3,
    RUBIK_L = 4,
    RUBIK_B = 5,
};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float solved;
    float distance;
    float scramble_depth;
    float curriculum_stage;
    float solve_rate_ema;
    float depth_n[RUBIK_LOG_DEPTHS];
    float depth_solved[RUBIK_LOG_DEPTHS];
    float depth_distance[RUBIK_LOG_DEPTHS];
    float depth_return[RUBIK_LOG_DEPTHS];
    float solved_return_n;
    float solved_return;
    float unsolved_return_n;
    float unsolved_return;
    float n;
} Log;

typedef struct {
    int sticker_size;
} Client;

typedef struct State {
    int step_count;
    int distance;
    int best_distance;
    int last_action;
    float episode_return;
    unsigned char cube[RUBIK_STICKERS];
} State;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    int max_steps;
    int scramble_depth;
    int curriculum_enabled;
    int curriculum_stage;
    int curriculum_stage_episodes;
    int curriculum_min_episodes;
    int curriculum_depth_0;
    int curriculum_depth_1;
    int curriculum_depth_2;
    int curriculum_depth_3;
    int fixed_mixed_enabled;
    int auto_reset_enabled;
    int fixed_mixed_count;
    int fixed_mixed_depths[RUBIK_MAX_FIXED_MIX];

    float step_penalty;
    float progress_reward;
    float solve_reward;
    float timeout_penalty;
    float curriculum_success_threshold;
    float curriculum_ema_alpha;
    float curriculum_success_ema;
    float fixed_depth_8_prob;
    float fixed_depth_9_prob;
    float fixed_depth_10_prob;
    float fixed_mixed_probs[RUBIK_MAX_FIXED_MIX];

    State state;
    Client* client;
} Rubik;

void c_reset(Rubik* env);
void c_step(Rubik* env);
void c_render(Rubik* env);
void c_close(Rubik* env);

static inline int rubik_min(int a, int b) {
    return a < b ? a : b;
}

static inline int rubik_max(int a, int b) {
    return a > b ? a : b;
}

static inline int rubik_clamp(int value, int lo, int hi) {
    return rubik_max(lo, rubik_min(value, hi));
}

static inline void rubik_set_solved(Rubik* env) {
    for (int face = 0; face < RUBIK_FACES; face++) {
        for (int i = 0; i < RUBIK_FACE_STICKERS; i++) {
            env->state.cube[face * RUBIK_FACE_STICKERS + i] = (unsigned char)face;
        }
    }
}

static inline void rubik_compute_observations(Rubik* env) {
    memset(env->observations, 0, RUBIK_OBS_SIZE * sizeof(unsigned char));
    for (int i = 0; i < RUBIK_STICKERS; i++) {
        unsigned char color = env->state.cube[i];
        if (color < RUBIK_FACES) {
            env->observations[i * RUBIK_FACES + color] = 1;
        }
    }
}

static inline int rubik_distance(Rubik* env) {
    int wrong = 0;
    for (int face = 0; face < RUBIK_FACES; face++) {
        for (int i = 0; i < RUBIK_FACE_STICKERS; i++) {
            int idx = face * RUBIK_FACE_STICKERS + i;
            wrong += env->state.cube[idx] != (unsigned char)face;
        }
    }
    return wrong;
}

static inline bool rubik_is_solved(Rubik* env) {
    return rubik_distance(env) == 0;
}

static inline int rubik_curriculum_depth(Rubik* env) {
    if (!env->curriculum_enabled) {
        return env->scramble_depth;
    }
    if (env->curriculum_stage == 0) {
        return env->curriculum_depth_0;
    } else if (env->curriculum_stage == 1) {
        return env->curriculum_depth_1;
    } else if (env->curriculum_stage == 2) {
        return env->curriculum_depth_2;
    }
    return env->curriculum_depth_3;
}

static inline void rubik_refresh_curriculum_depth(Rubik* env) {
    if (env->curriculum_enabled) {
        env->scramble_depth = rubik_curriculum_depth(env);
    }
}

static inline int rubik_sample_fixed_mixed_depth(Rubik* env) {
    if (env->fixed_mixed_count > 0) {
        float total = 0.0f;
        for (int i = 0; i < env->fixed_mixed_count && i < RUBIK_MAX_FIXED_MIX; i++) {
            if (env->fixed_mixed_depths[i] >= 0 && env->fixed_mixed_probs[i] > 0.0f) {
                total += env->fixed_mixed_probs[i];
            }
        }

        if (total <= 0.0f) {
            return env->scramble_depth;
        }

        float u = ((float)rand_r(&env->rng) / (float)RAND_MAX) * total;
        float cumulative = 0.0f;
        for (int i = 0; i < env->fixed_mixed_count && i < RUBIK_MAX_FIXED_MIX; i++) {
            if (env->fixed_mixed_depths[i] < 0 || env->fixed_mixed_probs[i] <= 0.0f) {
                continue;
            }
            cumulative += env->fixed_mixed_probs[i];
            if (u < cumulative) {
                return env->fixed_mixed_depths[i];
            }
        }
        int last_idx = rubik_min(env->fixed_mixed_count, RUBIK_MAX_FIXED_MIX) - 1;
        return env->fixed_mixed_depths[last_idx];
    }

    float p8 = env->fixed_depth_8_prob;
    float p9 = env->fixed_depth_9_prob;
    float p10 = env->fixed_depth_10_prob;
    float total = p8 + p9 + p10;

    if (total <= 0.0f) {
        return env->scramble_depth;
    }

    float u = ((float)rand_r(&env->rng) / (float)RAND_MAX) * total;
    if (u < p8) {
        return 8;
    }
    if (u < p8 + p9) {
        return 9;
    }
    return 10;
}

static inline void rubik_select_episode_depth(Rubik* env) {
    if (env->fixed_mixed_enabled) {
        env->scramble_depth = rubik_sample_fixed_mixed_depth(env);
    } else {
        rubik_refresh_curriculum_depth(env);
    }
}

static inline void rubik_update_curriculum(Rubik* env, int solved) {
    if (!env->curriculum_enabled) {
        return;
    }

    float alpha = env->curriculum_ema_alpha;
    if (alpha <= 0.0f || alpha > 1.0f) {
        alpha = 0.05f;
    }
    env->curriculum_success_ema =
        (1.0f - alpha) * env->curriculum_success_ema + alpha * (solved ? 1.0f : 0.0f);
    env->curriculum_stage_episodes += 1;

    if (env->curriculum_stage >= 3) {
        return;
    }

    if (env->curriculum_stage_episodes >= env->curriculum_min_episodes
            && env->curriculum_success_ema >= env->curriculum_success_threshold) {
        env->curriculum_stage += 1;
        env->curriculum_stage_episodes = 0;
        env->curriculum_success_ema = 0.0f;
        rubik_refresh_curriculum_depth(env);
    }
}

static inline void rubik_sticker_to_coord(
        int idx, int* x, int* y, int* z, int* nx, int* ny, int* nz) {
    int face = idx / RUBIK_FACE_STICKERS;
    int pos = idx % RUBIK_FACE_STICKERS;
    int row = pos / 3;
    int col = pos % 3;

    switch (face) {
    case RUBIK_U:
        *x = col - 1; *y = 1; *z = row - 1;
        *nx = 0; *ny = 1; *nz = 0;
        break;
    case RUBIK_R:
        *x = 1; *y = 1 - row; *z = 1 - col;
        *nx = 1; *ny = 0; *nz = 0;
        break;
    case RUBIK_F:
        *x = col - 1; *y = 1 - row; *z = 1;
        *nx = 0; *ny = 0; *nz = 1;
        break;
    case RUBIK_D:
        *x = col - 1; *y = -1; *z = 1 - row;
        *nx = 0; *ny = -1; *nz = 0;
        break;
    case RUBIK_L:
        *x = -1; *y = 1 - row; *z = col - 1;
        *nx = -1; *ny = 0; *nz = 0;
        break;
    default:
        *x = 1 - col; *y = 1 - row; *z = -1;
        *nx = 0; *ny = 0; *nz = -1;
        break;
    }
}

static inline int rubik_coord_to_sticker(int x, int y, int z, int nx, int ny, int nz) {
    int face;
    int row;
    int col;

    if (ny == 1) {
        face = RUBIK_U;
        row = z + 1;
        col = x + 1;
    } else if (ny == -1) {
        face = RUBIK_D;
        row = 1 - z;
        col = x + 1;
    } else if (nz == 1) {
        face = RUBIK_F;
        row = 1 - y;
        col = x + 1;
    } else if (nz == -1) {
        face = RUBIK_B;
        row = 1 - y;
        col = 1 - x;
    } else if (nx == 1) {
        face = RUBIK_R;
        row = 1 - y;
        col = 1 - z;
    } else {
        face = RUBIK_L;
        row = 1 - y;
        col = z + 1;
    }

    return face * RUBIK_FACE_STICKERS + row * 3 + col;
}

static inline void rubik_rotate_vec(int axis, int turns, int* x, int* y, int* z) {
    turns %= 4;
    if (turns < 0) {
        turns += 4;
    }

    for (int i = 0; i < turns; i++) {
        int ox = *x;
        int oy = *y;
        int oz = *z;

        if (axis == 0) {
            *x = ox;
            *y = -oz;
            *z = oy;
        } else if (axis == 1) {
            *x = oz;
            *y = oy;
            *z = -ox;
        } else {
            *x = -oy;
            *y = ox;
            *z = oz;
        }
    }
}

static inline void rubik_action_to_layer(int action, int* axis, int* layer, int* turns) {
    int face = action / 3;
    int turn = action % 3;

    switch (face) {
    case RUBIK_U: *axis = 1; *layer = 1; break;
    case RUBIK_R: *axis = 0; *layer = 1; break;
    case RUBIK_F: *axis = 2; *layer = 1; break;
    case RUBIK_D: *axis = 1; *layer = -1; break;
    case RUBIK_L: *axis = 0; *layer = -1; break;
    default: *axis = 2; *layer = -1; break;
    }

    if (turn == 0) {
        *turns = -*layer;
    } else if (turn == 1) {
        *turns = *layer;
    } else {
        *turns = 2;
    }
}

static inline void rubik_turn(Rubik* env, int action) {
    int axis;
    int layer;
    int turns;
    unsigned char next[RUBIK_STICKERS];
    memcpy(next, env->state.cube, RUBIK_STICKERS * sizeof(unsigned char));
    rubik_action_to_layer(action, &axis, &layer, &turns);

    for (int idx = 0; idx < RUBIK_STICKERS; idx++) {
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

static inline void rubik_scramble(Rubik* env) {
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
        rubik_turn(env, face * 3 + turn);
        last_face = face;
    }
}

static inline void rubik_start_episode(Rubik* env) {
    env->state.step_count = 0;
    env->state.episode_return = 0.0f;
    env->state.last_action = -1;
    rubik_scramble(env);
    env->state.distance = rubik_distance(env);
    env->state.best_distance = env->state.distance;
    rubik_compute_observations(env);
}

static inline void rubik_add_depth_log(Rubik* env, int solved) {
    float solved_f = solved ? 1.0f : 0.0f;
    float distance = (float)env->state.distance;
    float episode_return = env->state.episode_return;

    if (env->scramble_depth >= 0 && env->scramble_depth < RUBIK_LOG_DEPTHS) {
        env->log.depth_n[env->scramble_depth] += 1.0f;
        env->log.depth_solved[env->scramble_depth] += solved_f;
        env->log.depth_distance[env->scramble_depth] += distance;
        env->log.depth_return[env->scramble_depth] += episode_return;
    }
}

static inline void rubik_add_outcome_return_log(Rubik* env, int solved) {
    if (solved) {
        env->log.solved_return_n += 1.0f;
        env->log.solved_return += env->state.episode_return;
    } else {
        env->log.unsolved_return_n += 1.0f;
        env->log.unsolved_return += env->state.episode_return;
    }
}

static inline void rubik_add_log(Rubik* env, int solved) {
    env->log.perf += solved ? 1.0f : 0.0f;
    env->log.score += env->state.episode_return;
    env->log.episode_return += env->state.episode_return;
    env->log.episode_length += (float)env->state.step_count;
    env->log.solved += solved ? 1.0f : 0.0f;
    env->log.distance += (float)env->state.distance;
    env->log.scramble_depth += (float)env->scramble_depth;
    env->log.curriculum_stage += (float)env->curriculum_stage;
    env->log.solve_rate_ema += env->curriculum_success_ema;
    rubik_add_depth_log(env, solved);
    rubik_add_outcome_return_log(env, solved);
    env->log.n += 1.0f;
}

void c_reset(Rubik* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    rubik_start_episode(env);
}

void c_step(Rubik* env) {
    int action = (int)env->actions[0];
    env->terminals[0] = 0.0f;

    int prev_distance = env->state.distance;
    float reward = -env->step_penalty;

    if (action < 0 || action >= RUBIK_ACTIONS) {
        reward -= 0.25f;
    } else {
        rubik_turn(env, action);
        if (action == env->state.last_action) {
            reward -= 0.0025f;
        }
        env->state.last_action = action;
    }

    env->state.step_count += 1;
    env->state.distance = rubik_distance(env);
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
            rubik_start_episode(env);
        } else {
            rubik_compute_observations(env);
        }
    } else {
        rubik_compute_observations(env);
    }
}

static const Color RUBIK_COLORS[RUBIK_FACES] = {
    {245, 245, 245, 255},
    {190, 32, 38, 255},
    {26, 150, 65, 255},
    {245, 205, 38, 255},
    {237, 125, 49, 255},
    {35, 91, 168, 255},
};

static inline Client* rubik_make_client(void) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->sticker_size = 54;
    InitWindow(960, 720, "PufferLib Rubik");
    SetTargetFPS(30);
    return client;
}

static inline void rubik_draw_face(Rubik* env, int face, int x0, int y0, int sticker_size) {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int idx = face * RUBIK_FACE_STICKERS + row * 3 + col;
            int x = x0 + col * sticker_size;
            int y = y0 + row * sticker_size;
            Color color = RUBIK_COLORS[env->state.cube[idx] % RUBIK_FACES];
            DrawRectangle(x + 2, y + 2, sticker_size - 4, sticker_size - 4, color);
            DrawRectangleLines(x + 2, y + 2, sticker_size - 4, sticker_size - 4,
                (Color){8, 18, 22, 255});
        }
    }
}

static inline Vector2 rubik_v2_add(Vector2 a, Vector2 b) {
    return (Vector2){a.x + b.x, a.y + b.y};
}

static inline Vector2 rubik_v2_scale(Vector2 a, float s) {
    return (Vector2){a.x * s, a.y * s};
}

static inline void rubik_draw_quad(Vector2 a, Vector2 b, Vector2 c, Vector2 d, Color color) {
    DrawTriangle(a, b, c, color);
    DrawTriangle(a, c, d, color);
    DrawTriangle(a, c, b, color);
    DrawTriangle(a, d, c, color);
    DrawLineV(a, b, (Color){10, 18, 22, 255});
    DrawLineV(b, c, (Color){10, 18, 22, 255});
    DrawLineV(c, d, (Color){10, 18, 22, 255});
    DrawLineV(d, a, (Color){10, 18, 22, 255});
}

static inline int rubik_visible_sticker_index(int face, int row, int col) {
    if (face == RUBIK_U) {
        row = 2 - row;
    }
    return face * RUBIK_FACE_STICKERS + row * 3 + col;
}

static inline void rubik_draw_iso_face(
        Rubik* env, int face, Vector2 origin, Vector2 col_axis, Vector2 row_axis) {
    Vector2 inset_col = rubik_v2_scale(col_axis, 0.04f);
    Vector2 inset_row = rubik_v2_scale(row_axis, 0.04f);
    Vector2 cell_col = rubik_v2_scale(col_axis, 0.92f);
    Vector2 cell_row = rubik_v2_scale(row_axis, 0.92f);

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            Vector2 p = rubik_v2_add(origin, rubik_v2_add(
                rubik_v2_scale(col_axis, (float)col),
                rubik_v2_scale(row_axis, (float)row)));
            p = rubik_v2_add(p, rubik_v2_add(inset_col, inset_row));

            int idx = rubik_visible_sticker_index(face, row, col);
            Color color = RUBIK_COLORS[env->state.cube[idx] % RUBIK_FACES];
            Vector2 a = p;
            Vector2 b = rubik_v2_add(p, cell_col);
            Vector2 c = rubik_v2_add(rubik_v2_add(p, cell_col), cell_row);
            Vector2 d = rubik_v2_add(p, cell_row);
            rubik_draw_quad(a, b, c, d, color);
        }
    }
}

void c_render(Rubik* env) {
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        c_close(env);
        exit(0);
    }

    if (env->client == NULL) {
        env->client = rubik_make_client();
    }

    BeginDrawing();
    ClearBackground((Color){7, 13, 18, 255});

    Vector2 front = {280.0f, 250.0f};
    Vector2 x_axis = {66.0f, 0.0f};
    Vector2 y_axis = {0.0f, 66.0f};
    Vector2 z_axis = {42.0f, -32.0f};

    rubik_draw_iso_face(env, RUBIK_U, front, x_axis, z_axis);
    rubik_draw_iso_face(env, RUBIK_R, rubik_v2_add(front, rubik_v2_scale(x_axis, 3.0f)),
        z_axis, y_axis);
    rubik_draw_iso_face(env, RUBIK_F, front, x_axis, y_axis);

    DrawText(TextFormat("distance %d  step %d/%d  scramble %d  stage %d",
        env->state.distance, env->state.step_count, env->max_steps,
        env->scramble_depth, env->curriculum_stage),
        24, 24, 24, RAYWHITE);
    EndDrawing();
}

void c_close(Rubik* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}
