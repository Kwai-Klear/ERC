<div align="center">
<h1>✨ Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning</h1>
</div>


<div align="center">

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-red.svg)](https://arxiv.org/abs/2512.05591)
[![Issues](https://img.shields.io/badge/🐛_Issues-GitHub-blue.svg)](https://github.com/Kwai-Klear/ERC/issues)
[![Contact](https://img.shields.io/badge/📧_Contact-Email-green.svg)](mailto:suzhenpeng13@163.com)

</div>

## 📣 Latest News
**[December 5, 2025]** 🔍 We propose **entropy ratio clipping​** (ERC) to impose a global constraint on the output distribution of the policy model. Experiments demonstrate that ERC can significantly improve the stability of off-policy training. 📄 The paper is available on [arXiv](https://arxiv.org/abs/2512.05591).

**[September 26, 2025]** 🔍 We further explored GPPO in depth and proposed **CE-GPPO**, focusing on the impact of ppo-clip tokens on entropy. 📄 The paper is available on [arXiv](https://arxiv.org/pdf/2509.20712) and [HuggingFace Daily](https://huggingface.co/papers/2509.20712).

**[September 15, 2025]** GPPO brings benefits in others' industrial scenarios. 💼✨ Check the [Xiaohongshu link](http://xhslink.com/o/3MS0x3zuMix ). 🔗

**[August 12, 2025]** 🚀 We released the checkpoint for [KlearReasoner-8B](https://huggingface.co/Kwai-Klear/Klear-Reasoner-8B), along with the training data.

**[August 11, 2025]** 🔬 KlearReasoner-8B conducted preliminary exploration of GPPO.

**[August 11, 2025]** 🏆 We released KlearReasoner-8B, achieving SOTA performance among small-scale 7/8B models.

**[August 11, 2025]** 📢 KlearReasoner is available on [arXiv](https://arxiv.org/pdf/2508.07629) and [HuggingFace Daily](https://huggingface.co/papers/2508.07629).



## 💡 Motivation

Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, leading to training instabilities manifested as fluctuations in policy entropy and unstable gradients. Although PPO-Clip mitigates this issue through importance clipping, it still overlooks the global distributional shift of actions. To address these challenges, we propose using the entropy ratio between the current and previous policies as a new global metric that effectively quantifies the relative change in policy exploration throughout updates. Building on this metric, we introduce an **Entropy Ratio Clipping** (**ERC**) mechanism that imposes bidirectional constraints on the entropy ratio. This stabilizes policy updates at the global distribution level and compensates for the inability of PPO-clip to regulate probability shifts of un-sampled actions. We integrate ERC into both DAPO and GPPO reinforcement learning algorithms. Experiments across multiple benchmarks show that ERC consistently improves performance.


## Implementation of ERC

The complete loss implementation is as follows:
```python
@register_policy_loss("grpo_erc")
def compute_policy_loss_grpo_erc(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    old_entropy: torch.Tensor,
    entropy: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        pg_losses = pg_losses * tis_imp_ratio

    entropy_ratio = (entropy + 1e-8) / (old_entropy + 1e-8)
    # balanced_mask =  ((config.entropy_mask_low < entropy_ratio & entropy_ratio < config.entropy_mask_high) * response_mask.bool()).bool()
    balanced_mask = (
        ((entropy_ratio > config.entropy_mask_low) & (entropy_ratio < config.entropy_mask_high))
        & response_mask.bool()
    ).bool()

    pg_losses = torch.where(balanced_mask.bool(), pg_losses, pg_losses.detach())

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    entropy_frac = (
        ((entropy_ratio > config.entropy_mask_low) & (entropy_ratio < config.entropy_mask_high))
        & response_mask.bool()
    ).sum() / response_mask.bool().sum()
    
    return pg_loss, pg_clipfrac, ppo_kl, entropy_frac
```

---

## 🧪 Training
### Configure the experimental environment
```bash
git clone https://github.com/Kwai-Klear/ERC
cd ERC
pip install -e .
pip install -r requirements.txt
```
For mathematics, we use [math_verify](https://github.com/huggingface/Math-Verify) for judging.

### Download a pre-trained checkpoint & data
We trained our model based on [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) and [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), using the [KlearReasoner-MathSub-30K](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-MathSub-30K) dataset for training, with [AIME2024](https://github.com/Kwai-Klear/CE-GPPO/blob/main/benchmarks/aime960_math_verify.json) and [AIME2025](https://github.com/Kwai-Klear/CE-GPPO/blob/main/benchmarks/aime960_math_verify25.json) as the validation sets.

### Using Ray for Multi-Node Training
For multi-node training​​, ensure ​​all nodes are started and connected via Ray​​ before executing the training script. Below is a brief setup guide for Ray across multiple machines:
#### Step 1: Start Ray on the Head Node (node0)

On the first node (typically called `node0`), run:

```bash
ray start --head --dashboard-host=0.0.0.0
```

Get the IP address of the master node.
```bash
MASTER_IP=$(hostname -I | awk '{print $1}')
```
#### Step 2: Connect Other Nodes (e.g., node1)

On each additional worker node (e.g., `node1`), run the following, replacing the IP with that of your head node:

```bash
ray start --address=\"$MASTER_IP:6379\"
```

### RL Training
Run the following script on the master node to start the training task.


```bash
bash recipe/dapo/perf_run_dapo_ours_math_szp_ae_dapo_erc_1_05.sh # 7B + DAPO
bash recipe/dapo/perf_run_dapo_ours_code.sh # 7B + GPPO 
```

In the startup script, you need to set the following variables:
```bash
YOUR_MODEL_PATH="<your_model_path>"
CKPTS_SAVE_DIR="<ckpts_save_path>"
YOUR_TRAIN_FILE="<train_data_path>"
YOUR_TEST_FILE="<test_data_path>"
```


---
## 🤝 Citation
If you find this work helpful, please cite our paper:

```bibtex
@article{su2025entropy,
  title={Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning},
  author={Su, Zhenpeng and Pan, Leiyu and Lv, Minxuan and Mei, Tiehua and Lin, Zijia and Li, Yuntao and Hu, Wenping and Tang, Ruiming and Gai, Kun and Zhou, Guorui},
  journal={arXiv preprint arXiv:2512.05591},
  year={2025}
}
```


```bibtex
@misc{su2025cegppocontrollingentropygradientpreserving,
      title={CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning}, 
      author={Zhenpeng Su and Leiyu Pan and Minxuan Lv and Yuntao Li and Wenping Hu and Fuzheng Zhang and Kun Gai and Guorui Zhou},
      year={2025},
      eprint={2509.20712},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.20712}, 
}
```


```bibtex
@article{DBLP:journals/corr/abs-2508-07629,
  author       = {Zhenpeng Su and
                  Leiyu Pan and
                  Xue Bai and
                  Dening Liu and
                  Guanting Dong and
                  Jiaming Huang and
                  Wenping Hu and
                  Fuzheng Zhang and
                  Kun Gai and
                  Guorui Zhou},
  title        = {Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving
                  Clipping Policy Optimization},
  journal      = {CoRR},
  volume       = {abs/2508.07629},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2508.07629},
  doi          = {10.48550/ARXIV.2508.07629},
  eprinttype    = {arXiv},
  eprint       = {2508.07629},
  timestamp    = {Sat, 13 Sep 2025 14:46:27 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2508-07629.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


