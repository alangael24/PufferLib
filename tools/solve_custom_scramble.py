import argparse
import sys
import time

import torch

import pufferlib.pufferl as pufferl
from pufferlib import _C, torch_pufferl
from tools.render_rubik_policy import native_bin_to_torch_state, choose_action


FACE_TO_IDX = {
    "U": 0,
    "R": 1,
    "F": 2,
    "D": 3,
    "L": 4,
    "B": 5,
}


def parse_scramble(scramble):
    moves = []
    i = 0
    while i < len(scramble):
        ch = scramble[i].upper()
        if ch in " ,\t\n":
            i += 1
            continue
        if ch not in FACE_TO_IDX:
            raise ValueError(f"Bad move at position {i}: {scramble[i]!r}")
        suffix = ""
        if i + 1 < len(scramble) and scramble[i + 1] in ("'", "2"):
            suffix = scramble[i + 1]
            i += 1
        turn = 2 if suffix == "2" else 1 if suffix == "'" else 0
        moves.append((ch + suffix, FACE_TO_IDX[ch] * 3 + turn))
        i += 1
    return moves


def load_args(max_steps):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = pufferl.load_config("rubik")
    finally:
        sys.argv = argv
    args["slowly"] = True
    args["reset_state"] = False
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = 1
    args["train"]["total_timesteps"] = 1
    args["env"]["max_steps"] = max_steps
    args["env"]["auto_reset_enabled"] = 0
    args["env"]["fixed_mixed_enabled"] = 1
    args["env"]["fixed_mixed_count"] = 1
    args["env"]["fixed_depth_0"] = 0
    args["env"]["fixed_prob_0"] = 1.0
    for i in range(1, 8):
        args["env"][f"fixed_depth_{i}"] = 0
        args["env"][f"fixed_prob_{i}"] = 0.0
    return args


def step_action(vec, action_id):
    action = torch.tensor([[float(action_id)]], dtype=torch.float32).contiguous()
    vec.cpu_step(action.data_ptr())
    return torch_pufferl._cpu_tensor(vec.terminals_ptr, (1,), torch.float32)[0].item() > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scramble", nargs="*", help="Example: R U R' F2 L")
    parser.add_argument("--checkpoint", default="downloads/rubik_depth20_probe/best_depth20_probe.bin")
    parser.add_argument("--solve-budget", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.30)
    parser.add_argument("--greedy", action="store_true")
    args_cli = parser.parse_args()

    scramble = " ".join(args_cli.scramble).strip()
    if not scramble:
        scramble = input("scramble> ").strip()
    moves = parse_scramble(scramble)
    max_steps = len(moves) + args_cli.solve_budget

    args = load_args(max_steps)
    vec = _C.create_vec(args, 0)
    vec.reset()
    vec.log()

    policy = torch_pufferl.load_policy(args, vec)
    native_bin_to_torch_state(args_cli.checkpoint, policy)
    policy.eval()

    print(f"Scramble ({len(moves)} moves): {' '.join(name for name, _ in moves)}")
    print("Showing scramble, then policy solve. Close the window or press Esc to stop.")
    try:
        for name, action_id in moves:
            vec.render(0)
            time.sleep(args_cli.delay)
            step_action(vec, action_id)
            print(f"scramble move: {name}")

        vec.log()
        state = policy.initial_state(1, "cpu")
        sampled = not args_cli.greedy

        for solve_step in range(1, args_cli.solve_budget + 1):
            vec.render(0)
            time.sleep(args_cli.delay)
            obs = torch_pufferl._cpu_tensor(vec.obs_ptr, (1, vec.obs_size), torch.uint8)
            with torch.no_grad():
                logits, _, state = policy.forward_eval(obs, state)
                action = choose_action(logits, sampled)
            terminal = step_action(vec, int(action[0, 0].item()))
            if terminal:
                vec.render(0)
                logs = vec.log()
                solved = logs.get("solved", 0.0) > 0.5
                print(f"terminal at solve_step={solve_step} solved={solved}")
                time.sleep(max(2.0, args_cli.delay * 8))
                break
        else:
            print("solve budget exhausted")
            time.sleep(2.0)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
