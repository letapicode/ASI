import torch
from asi.self_play_skill_loop import SelfPlaySkillLoopConfig
from asi.world_model_rl import RLBridgeConfig, train_with_self_play


def main() -> None:
    sp_cfg = SelfPlaySkillLoopConfig(cycles=2, steps=5)
    rl_cfg = RLBridgeConfig(
        state_dim=sp_cfg.env_state_dim,
        action_dim=sp_cfg.action_dim,
        epochs=1,
        batch_size=4,
    )
    frames = [torch.randn(sp_cfg.img_channels, 8, 8) for _ in range(4)]
    actions = [0 for _ in frames]
    policy = lambda obs: torch.zeros((), dtype=torch.long)

    wm, skill_model = train_with_self_play(rl_cfg, sp_cfg, policy, frames, actions)
    print("world model trained", isinstance(wm, torch.nn.Module))
    print("skill model", type(skill_model).__name__)


if __name__ == "__main__":
    main()
