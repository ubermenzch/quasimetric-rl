# Remote Codex prompts for the GO-QRL Hybrid scale sweep

## Server 2x: 65 runs

```text
在本机 quasimetric-rl 仓库中添加 GO-QRL Max4 Hybrid 模型规模实验的
server_2x 分区。先检查 git status，保留所有已有改动，不启动或停止队列
runner，也不要覆盖现有任务。

用户已经将模型规模实现、任务生成器和清单上传到 origin/main。工作树允许
快进时先运行git pull --ff-only origin main；如果存在本地改动，先判断如何
安全保留并合入上游，禁止使用reset --hard或checkout丢弃改动。同步后确认
tools/generate_go_qrl_hybrid_scale_tasks.py和server_2x清单存在。

首先确认仓库已包含 GO-QRL M/L/XL/XXL/XXXL 五档：M 保持旧普通 MLP；
L/XL/XXL/XXXL 为宽度1024、latent512、逻辑深度4/8/16/32的残差MLP，
每4层一个残差块，每个隐藏Dense后为LayerNorm->SiLU，最终Linear后无LN、
激活或L2归一化；Encoder、Projector、Dynamics和Actor均按该深度扩展。
IQE从L起同步扩大：M/L为2048/64，XL为4096/128，XXL为8192/256，
XXXL为16384/512，始终保持每个component为32维。
如果这些实现或配置缺失，先按上述定义补齐并运行模型规模测试。

本服务器只负责以下13个(environment, scale)五seed组，共65个run：
- M: swimmer6, Pusher-v4, AntNavigate-v4
- L: manipulator_bring_ball, Pusher-v4, AntNavigate-v4
- XL: manipulator_bring_ball, FetchSlide, AntNavigate-v4
- XXL: swimmer6
- XXXL: FetchSlide, swimmer6, Pusher-v4

每组训练seed为1000-1004。每个run训练500000步，batch_size=256，
save_steps=50000，eval_steps=null，keep_only_latest_checkpoint=false。
每个checkpoint使用validation seed 1000开始评估500 episode，即1000-1499；
训练完成后按现有在线流程选validation最优checkpoint，并用test seed 1500开始
评估1000 episode，即1500-2499。

算法固定为GO-QRL Max4 Hybrid：joint training，separate_latent_dynamics=false，
latent dynamics distance=iqe_mse，mse_weight=1，iqe_weight=1；latent goal使用
mode=max、steps=4、direct search、SGD lr=0.01、keep_best=true；adaptive entropy
开启且entropy_mc_samples=100，add_goal_as_future_state=true，BC weight=0，
goal_set_distance关闭，exploration_eps=0，保留replay并启用resume_if_possible。
不要额外覆盖Projector activation，交给各模型规模preset决定。

环境映射：FetchSlide用env.kind=gcrl；swimmer6和manipulator_bring_ball用dmc；
Pusher-v4和AntNavigate-v4用gym_mujoco。

若仓库已有tools/generate_go_qrl_hybrid_scale_tasks.py，运行：
.venv/bin/python -m unittest tests.test_go_qrl_hybrid_scale_tasks
.venv/bin/python tools/generate_go_qrl_hybrid_scale_tasks.py --partition server_2x --append-to runs/qrl_queue/tasks.tsv
再次运行追加命令必须显示Added 0 tasks。最终确认恰好新增65项、task ID唯一，
并报告清单路径和检查结果。若生成器不存在，则按以上定义创建等价、可重复、
幂等的生成器后再追加，只能添加本分区，不能把all分区加入队列。
```

## Server 1x: 30 runs

```text
在本机 quasimetric-rl 仓库中添加 GO-QRL Max4 Hybrid 模型规模实验的
server_1x 分区。先检查 git status，保留所有已有改动，不启动或停止队列
runner，也不要覆盖现有任务。

用户已经将模型规模实现、任务生成器和清单上传到 origin/main。工作树允许
快进时先运行git pull --ff-only origin main；如果存在本地改动，先判断如何
安全保留并合入上游，禁止使用reset --hard或checkout丢弃改动。同步后确认
tools/generate_go_qrl_hybrid_scale_tasks.py和server_1x清单存在。

首先确认仓库已包含 GO-QRL M/L/XL/XXL/XXXL 五档：M 保持旧普通 MLP；
L/XL/XXL/XXXL 为宽度1024、latent512、逻辑深度4/8/16/32的残差MLP，
每4层一个残差块，每个隐藏Dense后为LayerNorm->SiLU，最终Linear后无LN、
激活或L2归一化；Encoder、Projector、Dynamics和Actor均按该深度扩展。
IQE从L起同步扩大：M/L为2048/64，XL为4096/128，XXL为8192/256，
XXXL为16384/512，始终保持每个component为32维。
如果这些实现或配置缺失，先按上述定义补齐并运行模型规模测试。

本服务器只负责以下6个(environment, scale)五seed组，共30个run：
- M: FetchSlide
- L: swimmer6
- XL: Pusher-v4
- XXL: FetchSlide, AntNavigate-v4
- XXXL: manipulator_bring_ball

每组训练seed为1000-1004。每个run训练500000步，batch_size=256，
save_steps=50000，eval_steps=null，keep_only_latest_checkpoint=false。
每个checkpoint使用validation seed 1000开始评估500 episode，即1000-1499；
训练完成后按现有在线流程选validation最优checkpoint，并用test seed 1500开始
评估1000 episode，即1500-2499。

算法固定为GO-QRL Max4 Hybrid：joint training，separate_latent_dynamics=false，
latent dynamics distance=iqe_mse，mse_weight=1，iqe_weight=1；latent goal使用
mode=max、steps=4、direct search、SGD lr=0.01、keep_best=true；adaptive entropy
开启且entropy_mc_samples=100，add_goal_as_future_state=true，BC weight=0，
goal_set_distance关闭，exploration_eps=0，保留replay并启用resume_if_possible。
不要额外覆盖Projector activation，交给各模型规模preset决定。

环境映射：FetchSlide用env.kind=gcrl；swimmer6和manipulator_bring_ball用dmc；
Pusher-v4和AntNavigate-v4用gym_mujoco。

若仓库已有tools/generate_go_qrl_hybrid_scale_tasks.py，运行：
.venv/bin/python -m unittest tests.test_go_qrl_hybrid_scale_tasks
.venv/bin/python tools/generate_go_qrl_hybrid_scale_tasks.py --partition server_1x --append-to runs/qrl_queue/tasks.tsv
再次运行追加命令必须显示Added 0 tasks。最终确认恰好新增30项、task ID唯一，
并报告清单路径和检查结果。若生成器不存在，则按以上定义创建等价、可重复、
幂等的生成器后再追加，只能添加本分区，不能把all分区加入队列。
```
