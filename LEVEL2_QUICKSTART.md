# Level 2 快速仿真

本目录已经选择 `fast_racing_controller.py` 和 `attitude` 控制模式。完整模型及 MIT 许可证在 `lsy_drone_racing/control/policies/`。

## 直接查看场景

在 VS Code 打开本目录，按 **Ctrl + Shift + B**。默认任务 `Level 2: visualize 20 races` 会通过本项目现有的 Pixi Python 环境启动官方仿真脚本，连续运行 20 次。

第一次会编译 JAX，需等待片刻；随后弹出的是 MuJoCo 三维仿真窗口。Pixi 是环境管理工具，没有单独的模拟场景窗口。

任务直接调用 `.pixi/envs/default/python.exe`，无需重新下载环境，也不需要为这个 RL 策略重新编译 acados。

你也可以在本目录的 PowerShell 终端手动运行：

```powershell
$env:SCIPY_ARRAY_API = '1'
pixi run python scripts/sim.py --config=level2.toml --render=True --n_runs=20
```

如果本机 Pixi 命令报告用户目录或激活脚本错误，使用已经安装的同一个环境：

```powershell
$env:SCIPY_ARRAY_API = '1'
.\.pixi\envs\default\python.exe scripts/sim.py --config=level2.toml --render=True --n_runs=20
```

在 VS Code 的运行任务终端按 **Ctrl + C** 停止。仿真脚本没有更改，个别随机圈仍会失败；这不是保证每圈完赛的策略。

## 查看正式成绩

选择“终端 → 运行任务 → Level 2: official evaluation”。它运行老师原版 `scripts/evaluate.py`，结束后成绩写入 `evaluation.csv`。

移植前验证：正式评测 17/20 成功、平均成功圈 3.351 秒；独立 100 次测试 81/100 成功、平均 3.360 秒。新一轮随机结果会有变化。这仅针对 Level 2。

公开策略的原作者、版本和复用说明见 `docs/control/fast_racing.md`。旧控制器文件保留；配置备份位于工作区 `outputs/leaderboard_research/vscode-update-backup/level2.toml.before`。
