# A5 → B：固定候选后的独立验证

这是 A 第一门专项训练完成后的接入节点。A 已完成模型、更新器、导出、部署与开发筛选；B 不需要重写 A 算法。

## 固定的三个对象

|对象|入口|模型说明|
|---|---|---|
|C0 基线|fast_racing_controller.py|原冻结参考模型|
|C3 对照|team_c3_pilot_controller.py|上一轮 stage 0/2 残差|
|A4 候选|team_a4_first_gate_controller.py|本轮第一门专用残差，stage 1..5 执行 C0|

A4 权重 SHA256：`13678eca8dc97c910307dfbcf27ad1998adbfd0045af0a1ae891d45559245618`。

老师保护的环境、物理、随机化、正式奖励、频率和比赛流程未变；正式配置允许项仍只有 controller.file、env.control_mode。本轮保持 attitude，不使用电机转速或将 4 维动作拼接到 16 维接口。默认配置也未修改。

本轮 A4 开发成绩为 161/200、成功均时 3.340 秒；C3 为 157/200、3.341274 秒。应表述为基本同速、开发集多完成 4 回合，尚未证明泛化提升。

## B 开始操作

完整包内的 bundle 保留完整 Git 历史及本地 A 分支。为避免覆盖你已有工作，先在新目录克隆：

```bash
git clone -b dev/policy-a-first-gate A5-source.bundle project
cd project
pixi install -e tests
pixi run -e tests python -m pytest tests/team_a tests/team_b tests/unit/test_fast_racing.py -q
pixi run -e tests python -m validation.team_b.preflight
```

运行记录、训练检查点及历史轨迹在交付包 records 中。预检只判断本地代码和资产就绪；验证是否已运行应查对应评测 manifest，不能仅凭源码推断。

## 1. 先复现开发集

在同一台电脑、同一个 Pixi 环境里依次评测三个对象，不与训练进程并行。使用新的输出目录，不覆盖原记录。

```bash
pixi run -e tests python -m validation.team_b.paired_eval --controller fast_racing_controller.py --seed 110001 --runs 200 --run-id B-C0-development --output-root ../B-results
pixi run -e tests python -m validation.team_b.paired_eval --controller team_c3_pilot_controller.py --seed 110001 --runs 200 --run-id B-C3-development --output-root ../B-results
pixi run -e tests python -m validation.team_b.paired_eval --controller team_a4_first_gate_controller.py --seed 110001 --runs 200 --run-id B-A4-development --output-root ../B-results
```

若成绩不能接近交付记录，先核对 SHA、配置哈希、JAX/crazyflow 等版本和解释器。不要靠修改物理参数“复现”分数。跨平台不要求逐浮点位完全一致，要保留环境差异及逐场记录。

## 2. 开启本轮保留验证集，禁止改权重

确认复现后，B 对以上三个固定对象依次使用 `--split validation --seed 120001 --runs 100`，run-id 分别使用 B-C0-validation、B-C3-validation、B-A4-validation。其余参数保持一致。

这是 A4 固定后的首次保留验证。验证种子不得加入训练；A 不得根据这些种子逐个调控制器。最终集 130001–130500 仍不打开。

评测器会记录 split=validation，并拒绝把验证种子作为 development 使用；最终集仍由程序拒绝打开。

## 3. 交回 A 的内容

- 三组原始 episodes.jsonl、summary.json、manifest.json，以及相同场景的成功/失败变化。
- 成功次数、95% 区间、共同成功回合时间差、第一门及后续失败分布、20ms 超时。
- 在不知道验证结果前固定接受规则：A4 成功数不低于 C3、成功均时不高于 C3、控制无超时，且至少一项性能严格改进，才考虑进入最终冻结评测。
- 该规则只是本轮筛选；100 场不足以保证小幅提升的统计确定性。若没有通过，保留基线，并反馈失败类型，不上传 main。

若本轮未通过，再由 B 在独立训练池采集门位置更新幅度、首次可用观测时的距离/速度、剩余控制步等公共观测事件，扩展难例课程；A 可继续使用相同模型 API。

## 再训练的 A 入口

```bash
pixi run -e tests python -m training.team_a.train_first_gate --seed 20261029 --updates 32 --output-dir ../new-training-run
```

训练入口会拒绝覆盖目录。恢复训练时同时给 `--resume 已有/checkpoint.pt`、更大的目标 `--updates` 和全新输出目录。不要把开发或验证种子放入训练池。

源码、模型、正式配置均已就绪；本轮交付不需要变更 4 维接口。不得直接更新 main 或将保留集评测当成课程正式提交。
