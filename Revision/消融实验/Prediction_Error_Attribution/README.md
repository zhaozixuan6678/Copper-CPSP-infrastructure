# Prediction Error Reduction Attribution

这套代码允许 Stage 0、Stage 1、Stage 2 分别来自不同 Excel 文件、不同 Sheet、不同列布局和不同特征集合。它不会依赖三个表的行顺序，而是先规范化 `ID`，取三个 Stage 的共同有效 ID，并为每个性能生成唯一、固定的 80/20 Train/Test ID 划分。默认启用 `cumulative_features=true`，因此后续 Stage 会保留前面 Stage 的全部特征，只加入当前表提供的新特征。

## 当前默认映射

- Stage 0: `FE/Feature1.xlsx`，仅使用原始成分列。
- Stage 1: `FE/Feature/Feature2.xlsx`，在 Stage 0 上累积传统成分/物理化学描述符；已明确排除 Ni/Si、乘积/比值和加工交互项，避免把 CTD 特征提前放入 Stage 1。
- Stage 2: `FE/Feature3.xlsx`，在 Stage 1 上累积表中除 ID 和目标列外的 CTD 特征。
- Stage 3: 不读取第四张表；复用 Stage 2 的特征和样本，仅在训练集内部做 5-fold CV 超参数搜索。

默认三个性能统一使用 XGBoost：HV = XGBoost；EC = XGBoost；Q3 = XGBoost。建议优先打开 `Prediction_Error_Attribution.ipynb` 分单元格执行；模型映射也可在 `attribution_config.json` 中修改。

Notebook 的第 1 节是手动输入区。`USER_STAGE_FILES` 决定三个文件的科学顺序，文件名中的 1/2/3 不会自动决定 Stage：`S0` 必须对应 composition-only，`S1` 对应 physicochemical descriptors，`S2` 对应 CTD-derived CPSP features。每个文件的 HV、EC、Q3 Sheet 名在 `USER_STAGE_SHEETS` 中独立指定，三个表不同的目标列名在 `USER_STAGE_TARGET_COLUMNS` 中指定。本次选择会写入 `runtime_config.json`，不覆盖基础模板。

所有可修改参数现已整理在同一个 `USER CONFIGURATION TRUNK` 代码单元格，包括路径、Sheet、目标列、模型、随机种子、测试集比例、交叉验证、搜索次数及运行开关。后续单元格不再包含需要手工修改的参数。

输入审计现支持自动解析。`AUTO_RESOLVE_INPUTS=True` 时，只需在 `CANDIDATE_FEATURE_FILES` 中列出候选 Excel；程序会依据 Sheet、目标列别名、Stage 必需特征、共同 ID、目标一致性和 CTD marker 自动选择唯一的 S0/S1/S2 组合，并输出 `auto_input_audit_report.csv` 与 `auto_resolved_config.json`。若没有合法组合或出现多个合法组合，程序停止并报告，不会静默猜测。

Stage 3 现支持 Optuna TPE 调优。默认 `STAGE3_OPTIMIZER='optuna'`、`OPTUNA_TRIALS=150`、5-fold training CV，并以 CV RMSE 为目标；`OPTUNA_TARGET_CV_R2=0.90` 仅作为训练集 CV 的可选提前停止标准，不使用测试集选择参数，也不保证所有性能达到0.90。正式结果同时保存 `CV_R2_training_only`、`CV_RMSE_training_only`、最终 Test R² 和 Test NRMSE。

## 执行顺序

先只检查输入、ID、目标值和特征列：

```bash
cd '/Users/zixuanzhao/Desktop/Corpus–Topic–Document/ML2/消融实验/Prediction_Error_Attribution'
python3 prediction_error_attribution.py --audit-only
```

快速试跑（Stage 3 仅搜索 3 组参数）：

```bash
python3 prediction_error_attribution.py --quick
```

正式运行（默认 Stage 3 搜索 30 组参数）：

```bash
python3 prediction_error_attribution.py
```

## 计算定义

主误差指标只使用测试集：

```text
E = Test NRMSE = RMSE(y_test, y_pred) / population SD(y_test)
```

误差降低归因：

```text
ΔPhysics = E0 - E1
ΔCTD     = E1 - E2
ΔModel   = E2 - E3
Residual    = E3
E0 = ΔPhysics + ΔCTD + ΔModel + Residual
```

不要使用 `0.8 × Train R² + 0.2 × Test R²` 计算 prediction error。Skew R² 可以作为附加模型表现指标，但不能代替独立测试集误差。

若某一步误差反而升高，对应贡献会是负值。代码会保留这个结果并给出警告，不能人为截断为零。

## 输出

正式结果位于 `results/`；`--quick` 的诊断结果单独位于 `results_quick/`，不会覆盖正式结果：

- `input_audit.csv`: 各 Stage 的源文件、Sheet、共同 ID 数和真实特征清单。
- `split_map.csv`: 每个性能的固定 Train/Test ID。
- `run_summary.csv`: S0–S3 的 Train/Test R²、RMSE、MAE、Test SD 和 Test NRMSE。
- `predictions_long.csv`: 每条记录的 ID、split、真实值和预测值。
- `best_parameters.csv`: Stage 3 在训练集 CV 中选出的参数。
- `attribution_summary.csv`: E0–E3、三类误差降低、Residual 及百分比。
- `prediction_error_attribution.png`: stacked-bar 图。

## 更换输入表

只编辑 `attribution_config.json`：

1. 修改每个 Stage 的 `file`；
2. 在 `sheets` 中分别声明 HV、EC、Q3 对应的 Sheet；
3. 在 `features` 中给出各性能的精确特征名；
4. 只有确认该 Sheet 的全部非 ID/目标列都是所需特征时，才使用 `__ALL_EXCEPT_ID_AND_TARGET__`。

代码会在列缺失、有效 ID 重复、三个 Stage 没有共同 ID或同一 ID 的目标值不一致时停止，防止产生不可比结果。
