# 项目进展记录

## 2026-03-11

### 当前阶段总结

- 任务是 AI6126 Project 1 的人脸语义分割 / face parsing。
- 核心约束是模型可训练参数量必须小于 `1,821,085`。
- 当前主力模型路线是 `deeplab`，不是最早 README 里写的 `attention_unet`。

### 已确认的实验结果

- `wolr_weighted_sqrt_d7c3_st20`：最佳 F-score 为 `0.7641`，出现在第 `48` 个 epoch
- `weighted_sqrt_d7c3_st20`：最佳 F-score 为 `0.7027261973759753`，出现在第 `26` 个 epoch
- `deeplab_d7c3`：最佳 F-score 为 `0.7022866198657503`，出现在第 `24` 个 epoch
- `deeplab_baseline`：最佳 F-score 为 `0.6937163352614634`，出现在第 `42` 个 epoch

补充说明：

- `best_model.pth` 不是按最后一个 epoch 保存，而是按验证集最高 `F-score` 保存。
- 上述最高分已经通过扫描所有已保存 `metrics.csv` 的历史峰值确认过。
- 当前已保存实验中，最高验证分数是 `0.7641`，目录为 `checkpoints/deeplab/wolr_weighted_sqrt_d7c3_st20/`。



## 2026-03-12 

### 数据集观察与结论

- 训练集 mask 明确区分左右语义部位。
- 目前已经确认至少包含：左眉 / 右眉、左眼 / 右眼、左耳 / 右耳。
- 从数据观察来看，整套数据在图像空间中的左右分布是固定的，也就是说图像左边和右边对应的语义类别具有稳定约定。
- 目前的判断是：这个作业不适合做水平翻转增强。

原因：

- 即使在翻转后同步交换左右类别，水平翻转仍然会人为引入镜像分布，而这类分布看起来并不符合当前训练集和测试集的实际规律。
- 既然训练集和测试集都遵循相同的左右布局先验，保留这种先验更稳妥。

当前结论：

- 后续训练不再使用 horizontal flip。

### 已完成

- 先从 [src/dataset.py](/Users/chenyixuan/Documents/NTU课程/ACV/AI6126_project1/src/dataset.py) 中移除水平翻转增强。
- 再将旋转增强中 `mask` 的填充值改为 `ignore_index=255`（之前默认会填成 `0` 背景类）。
- 修改后，旋转产生的外扩区域不会被当作真实背景参与训练损失计算。
- 85% 训练集开发实验：`去掉水平翻转` 版本已先提交一次，线上分数 `0.84`。最佳验证 `0.7641`，出现在第48个epoch。
- 85% 训练集开发实验：`去掉水平翻转 + 旋转 fill=255`（`wolr_fillrot_weighted_sqrt_d7c3_st20`）最佳验证 F-score 为 `0.7581`，出现在第 `45` 个 epoch。
- 线上提交结果：`wolr_fillrot_weighted_sqrt_d7c3_st2.zip`（2026-03-11 23:14）得分 `0.83`。
- 对比结论：当前不采用 `fill=255` 这一改动作为最终提交方案，最终全量训练优先使用“仅去掉水平翻转”的版本。
- 在当前最佳开发配置 `wolr_weighted_sqrt_d7c3_st20` 的基础上，额外加入 `boundary CE` 做单变量实验。
- 85% 训练集开发实验：`boundary CE` 版本最佳验证 F-score 为 `0.7539`，出现在第 `28` 个 epoch。
- 该实验方向可行，但当前配置下未超过 `wolr_weighted_sqrt_d7c3_st20` 的最佳验证 F-score `0.7641`。
- 当前判断：`boundary CE` 保留为候选方案，但暂不更新为新的本地最佳配置。
- 85% 训练集开发实验：`boundary CE` + `Dice:CE=5:5`（由 `7:3` 调整）最佳验证 F-score 为 `0.7597`，仍未超过当前本地最佳 `0.7641`。
- 结论：该配置不作为主线，模型文件不保留。
- 85% 训练集开发实验：`focal CE` 版本（`focal_wolr_weighted_sqrt_d7c3_st10`）最佳验证 F-score 为 `0.7612`，出现在第 `27` 个 epoch。
- 85% 训练集开发实验：`less crop` 版本（`lesscrop_wolr_weighted_sqrt_d7c3_st10`）最佳验证 F-score 为 `0.7597`，出现在第 `40` 个 epoch。
- 85% 训练集开发实验：`refine head` 版本（`refinehead_wolr_weighted_sqrt_d7c3_st10`）最佳验证 F-score 为 `0.7526`，出现在第 `29` 个 epoch。
- 对比结论：当前新增实验（`boundary CE`、`focal CE`、`less crop`、`refine head`）均未超过本地最佳 `0.7641`，主配置暂不变更。
- 85% 训练集开发实验：在当前主线配置上加入新的增强策略（`augmix_wolr_weighted_sqrt_d7c3_st10`），当前已达到最佳验证 F-score `0.7677`（超过此前本地最佳 `0.7641`）。
- 基于 `augmix_wolr_weighted_sqrt_d7c3_st10` 的 `best_model.pth` 已生成可提交压缩包：`submissions/augmix_wolr_weighted_sqrt_d7c3_st10.zip`（`masks/*.png`）。
- 训练框架已接入 `Lovasz-Softmax`（`ce_type=lovasz`），作为与 F-measure 更一致的 surrogate loss；下一步在 `augmix` 基线上开始对照实验。

### 最终训练与选模策略

- 最终提交阶段使用 `VAL_SPLIT=0.0` 在 100% 训练集上训练。
- 由于全量训练没有验证 F-score，不能用 `best_model.pth`（按验证分数）选模型。
- 基于 85% 开发实验最佳 epoch，按训练步数换算，100% 全量训练主候选设为 `epoch 41`。
- 同时考虑全量数据可能使最优点后移，在同一次训练中额外保存 `epoch 45` 作为备选。
- 执行方式是一次训练跑到 45 个 epoch，并通过 `SAVE_EPOCHS=41,45` 同时导出两个候选权重。
- 最终以线上提交得分对比 `epoch 41` 与 `epoch 45`，选更高者作为最终结果。

### 全量训练线上结果

- `epoch 41` 对应的线上提交得分为 `0.84`。
- `epoch 45` 对应的线上提交得分为 `0.83`。
- 结论：当前全量训练最终候选为 `epoch 41`，`epoch 45` 不再作为主候选继续使用。
- 目前 `0.84` 与此前最佳线上结果持平，说明当前主配置已经具备稳定复现能力。

### 类别映射核对（用于后处理）

- 已完成可视化核对，`id=3` 确认为“眼镜（glasses）”。
- 当前按项目数据核对得到的映射如下（按人物左右定义，括号内为画面左右说明）：
- `0`: background
- `1`: skin
- `2`: nose
- `3`: glasses
- `4`: left eye（画面右侧）
- `5`: right eye（画面左侧）
- `6`: left brow（画面右侧）
- `7`: right brow（画面左侧）
- `8`: left ear（画面右侧）
- `9`: right ear（画面左侧）
- `10`: mouth inner（上下唇之间）
- `11`: upper lip
- `12`: lower lip
- `13`: hair
- `14`: hat
- `15`: ear accessories
- `16`: necklace（待进一步确认）
- `17`: neck
- `18`: cloth
- 备注：后处理阶段优先只对 `hair/background/skin` 做碎块与边界平滑，避免误伤小器官类（眼睛、嘴唇、耳饰等）。

### 后处理压缩包实验记录

- 基线无后处理：线上 `0.84`（当前最佳，作为对照）。
- `mainonly_post.zip`：主块并类（默认全非背景类参与），线上结果 `0.83`（未提升）。
- `mainonly_no151618_post.zip`：主块并类，排除 `15/16/18`（耳饰/项链/衣服），已生成压缩包，待线上验证。
- `mainonly_relaxed_post.zip`：主块并类（仅 `3,4,5,6,7,8,9,10,11,12,14` 参与；`1/2/13/17` 允许多块），已生成压缩包，待线上验证。
- 结论更新：后处理目前“视觉更平滑”但尚未带来分数提升，暂不替代基线提交。

### TODO

- 明天（3月12日）重点处理头发区域误分问题，优先关注 `hair vs background` 边界质量。
- 观察到预测掩码存在明显碎斑与锯齿边缘（例如帽子区域被预测成一块一块的小区域），需要专项处理。
- 尝试推理后处理以减少碎斑噪声：小连通域过滤、轻量闭运算、局部多数投票滤波（先做可开关版本便于对照）。
- 对比“原始预测”与“后处理预测”的线上分数，确认后处理是否带来稳定增益。
- 训练侧改进方案 1（优先）：在现有 `CE + Dice` 上增加边界加权损失（边界像素更高权重），提升轮廓连续性并减少碎块。
- 训练侧改进方案 2：尝试 `Focal CE` 或 `OHEM CE`，重点优化帽子/背景中的困难负样本，抑制头发类误检小块。
- 训练侧改进方案 3：收紧增强强度（减小旋转角度、收紧随机裁剪范围），降低增强引入的伪边界噪声。
- 训练实验顺序：先只改方案 1 做单变量对照，确认有效后再叠加方案 2 或 3。
- 尝试边界损失方案 A（优先）：在现有 `CE + Dice` 上增加 `Boundary BCE/CE`，直接强化边界像素监督。
- 尝试边界损失方案 B（次优先）：实验 `Lovasz-Softmax`，对比其对发丝边界与整体分数的影响。
- 两个边界损失方案按 A -> B 顺序做对照实验，固定其余超参数，分别记录本地验证分数与线上提交分数。
- 把当前最佳提交候选使用的超参数整理到同一个地方，避免后面写报告时遗漏。
- 在选定的 conda 环境中完整测试 [run_predict_zip.sh](/Users/chenyixuan/Documents/NTU课程/ACV/AI6126_project1/run_predict_zip.sh)。
- 开始整理报告要点，包括：模型选择、loss 设计、数据增强策略，以及为什么移除 horizontal flip。
