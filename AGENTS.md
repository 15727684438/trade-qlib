# AGENTS — 项目续做入口

本项目为 **MNQ + qlib 量化研究**（工作目录内的 `qlib_research/`）。

当用户说 **"继续未完成的项目"**（或类似"续做/接着做"）时：

1. **先读** `qlib_research/HANDOFF.md` —— 它是完整交接文档，包含：目标与验收、关键决策/铁律、文件地图、数据重建、**过拟合审计结论**、当前待办与下一步。
2. 再读 `qlib_research/tf_mnq/config/t1_sealed_15min.yaml`（冻结配置+IS指标）与 `qlib_research/tf_mnq/results/audit_*.csv`（审计证据）。
3. 按 `HANDOFF.md` §7「待办」继续，**遵守 §3 铁律**（IS/OOS 纪律、成本口径、选择协议、$50k/1–2 手/MaxDD≤$1,500）。

关键结论（勿遗忘）：冻结 15min 策略**样本外不达标**（OOS PF 1.134 < 1.15；2025 PF≈1.007），且**样本内 1 手 MaxDD $1,736 已超 $1,500**。突破择时真实，4 条窄带否决过拟合。**下一步应重做止损/仓位/风险模型**，而非直接追回报。

环境：Python 3.11 + `pyqlib==0.9.7`；一键搭建见 `qlib_research/bootstrap.ps1`。
