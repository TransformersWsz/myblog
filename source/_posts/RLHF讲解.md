---
title: RLHF讲解
mathjax: true
toc: true
date: 2023-11-13 02:15:29
updated: 2023-11-13 02:15:29
categories:
- NLP
tags:
- LLM
- PPO
- RM
- Actor-Critic
---
RLHF包含了两个至关重要的步骤：
1. 训练Reward Model
2. 用Reward Model和SFT Model构造Reward Function，基于PPO算法来训练LLM
   1. frozen RM
   2. frozen SFT Model
   3. Actor $\pi_{\Phi}^{R L}$ initialized from SFT Model
   4. Critic $V_\eta$ initialized from RM

最大化目标函数：
$$
\begin{aligned}
\text { objective }(\phi)= & E_{(x, y) \sim D_{\pi_\phi \mathrm{RL}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]+ \\
& \gamma E_{x \sim D_{\text {pectrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
\end{aligned}
$$


<!--more-->

![rlhf](https://github.com/TransformersWsz/picx-images-hosting/raw/master/image.6qiivvmcc5c0.png)

训练流程：
```python
policy_model = load_model()
ref_policy_model = policy_model.copy()

for k in range(20000):
    # 采样（上一个epoch的actor模型和critic模型）
    prompts = sample_prompt()
    # old_log_probs是上一个epoch的actor模型的对数概率
    # old_values是上一个epoch的critic模型的预估期望收益
    responses, old_log_probs, old_values = respond(policy_model, prompts)

    # 反馈
    # 固定的reward模型
    scores = reward_model(prompts, responses)
    # 固定的sft模型
    ref_log_probs, _ = analyze_responses(ref_policy_model, prompts, responses)
    rewards = reward_func(reward_model, scores, old_log_probs, ref_log_probs)
    
    # 学习，为了更新actor和critic模型
    for epoch in range(4):
        # 这里的values用于更新critic模型
        log_probs, values = analyze_responses(policy_model, prompts, responses)
        advantages = advantage_func(rewards, old_values)
        actor_loss = actor_loss_func(advantages, old_log_probs, log_probs)
        critic_loss = critic_loss_func(rewards, values)
        loss = actor_loss + 0.1 * critic_loss
        train(loss, policy_model.parameters())
```
- frozen RM 和 frozen SFT是用来计算rewards的
- actor和critict会在epoch训练中同步更新
___

## 参考
- [RLHF理论篇](https://zhuanlan.zhihu.com/p/657490625)
- [拆解大语言模型RLHF中的PPO](https://zhuanlan.zhihu.com/p/645225982)