import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pufferlib.pufferl as pufferl
from pufferlib import _C, torch_pufferl


def configure_depth20(args, depth, max_steps):
    args["slowly"] = True
    args["reset_state"] = False
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = 1
    args["train"]["total_timesteps"] = 1
    args["env"]["max_steps"] = max_steps
    args["env"]["fixed_mixed_enabled"] = 1
    args["env"]["fixed_mixed_count"] = 1
    args["env"]["fixed_depth_0"] = depth
    args["env"]["fixed_prob_0"] = 1.0
    for i in range(1, 8):
        args["env"][f"fixed_depth_{i}"] = 0
        args["env"][f"fixed_prob_{i}"] = 0.0
    return args


def native_bin_to_torch_state(raw_path, policy):
    raw = np.fromfile(raw_path, dtype=np.float32)
    state = policy.state_dict()

    hidden = state["encoder.encoder.weight"].shape[0]
    obs = state["encoder.encoder.weight"].shape[1]
    actions = state["decoder.decoder.weight"].shape[0]
    layers = len([k for k in state if k.startswith("network.layers.")])

    expected = hidden * obs + (actions + 1) * hidden + layers * (3 * hidden * hidden)
    if raw.size != expected:
        raise RuntimeError(f"raw checkpoint has {raw.size} floats, expected {expected}")

    idx = 0
    new_state = {}

    n = hidden * obs
    new_state["encoder.encoder.weight"] = torch.from_numpy(
        raw[idx:idx + n].reshape(hidden, obs).copy())
    idx += n
    new_state["encoder.encoder.bias"] = torch.zeros_like(state["encoder.encoder.bias"])

    n = (actions + 1) * hidden
    fused_decoder = raw[idx:idx + n].reshape(actions + 1, hidden)
    idx += n
    new_state["decoder.decoder.weight"] = torch.from_numpy(fused_decoder[:actions].copy())
    new_state["decoder.decoder.bias"] = torch.zeros_like(state["decoder.decoder.bias"])
    new_state["decoder.value_function.weight"] = torch.from_numpy(
        fused_decoder[actions:actions + 1].copy())
    new_state["decoder.value_function.bias"] = torch.zeros_like(
        state["decoder.value_function.bias"])

    for layer in range(layers):
        key = f"network.layers.{layer}.weight"
        n = state[key].numel()
        new_state[key] = torch.from_numpy(raw[idx:idx + n].reshape(state[key].shape).copy())
        idx += n

    policy.load_state_dict(new_state)
    return new_state


def choose_action(logits, sampled):
    if sampled:
        action, _, _ = torch_pufferl.sample_logits(logits)
        return action.to(dtype=torch.float32).contiguous()
    if isinstance(logits, torch.Tensor):
        return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.float32).contiguous()
    raise RuntimeError("Rubik should expose one discrete logits tensor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/rubik_depth20_probe/best_depth20_probe.bin")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--delay", type=float, default=0.08)
    args_cli = parser.parse_args()

    checkpoint = Path(args_cli.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = pufferl.load_config("rubik")
    finally:
        sys.argv = argv
    args = configure_depth20(args, args_cli.depth, args_cli.max_steps)
    vec = _C.create_vec(args, 0)
    vec.reset()
    vec.log()
    policy = torch_pufferl.load_policy(args, vec)
    native_bin_to_torch_state(checkpoint, policy)
    policy.eval()

    state = policy.initial_state(1, "cpu")
    print(f"Rendering {checkpoint} at scramble_depth={args_cli.depth}")
    print("Close the Raylib window or press Esc to stop.")

    try:
        while True:
            vec.render(0)
            obs = torch_pufferl._cpu_tensor(vec.obs_ptr, (1, vec.obs_size), torch.uint8)
            with torch.no_grad():
                logits, _, state = policy.forward_eval(obs, state)
                action = choose_action(logits, args_cli.sampled)
            vec.cpu_step(action.data_ptr())
            terminals = torch_pufferl._cpu_tensor(vec.terminals_ptr, (1,), torch.float32)
            if terminals[0].item() > 0.0:
                state = policy.initial_state(1, "cpu")
            time.sleep(args_cli.delay)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
