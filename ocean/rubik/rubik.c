#include <stdio.h>
#include <time.h>
#include "rubik.h"

static Rubik* g_env = NULL;

static void demo_cleanup(void) {
    if (g_env == NULL) {
        return;
    }

    free(g_env->observations);
    free(g_env->actions);
    free(g_env->rewards);
    free(g_env->terminals);
    c_close(g_env);
    g_env = NULL;
}

static int key_action(void) {
    int face = -1;
    if (IsKeyPressed(KEY_U)) face = RUBIK_U;
    if (IsKeyPressed(KEY_R)) face = RUBIK_R;
    if (IsKeyPressed(KEY_F)) face = RUBIK_F;
    if (IsKeyPressed(KEY_D)) face = RUBIK_D;
    if (IsKeyPressed(KEY_L)) face = RUBIK_L;
    if (IsKeyPressed(KEY_B)) face = RUBIK_B;
    if (face < 0) {
        return -1;
    }

    int turn = 0;
    if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
        turn = 1;
    } else if (IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) {
        turn = 2;
    }
    return face * 3 + turn;
}

int demo(void) {
    Rubik env = {
        .num_agents = 1,
        .rng = (unsigned int)time(NULL),
        .max_steps = 40,
        .scramble_depth = 3,
        .step_penalty = 0.01f,
        .progress_reward = 0.03f,
        .solve_reward = 5.0f,
        .timeout_penalty = 1.0f,
    };

    g_env = &env;
    atexit(demo_cleanup);

    env.observations = (unsigned char*)calloc(RUBIK_OBS_SIZE, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);

    while (IsWindowReady() && !WindowShouldClose()) {
        int action = key_action();
        if (action >= 0) {
            env.actions[0] = (float)action;
            c_step(&env);
        } else if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = (float)(rand_r(&env.rng) % RUBIK_ACTIONS);
            c_step(&env);
        } else if (IsKeyPressed(KEY_MINUS)) {
            env.scramble_depth = rubik_max(0, env.scramble_depth - 1);
            c_reset(&env);
        } else if (IsKeyPressed(KEY_EQUAL)) {
            env.scramble_depth += 1;
            c_reset(&env);
        } else if (IsKeyPressed(KEY_BACKSPACE)) {
            c_reset(&env);
        }
        c_render(&env);
    }

    demo_cleanup();
    return 0;
}

int main(void) {
    return demo();
}
