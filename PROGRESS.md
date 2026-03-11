# 项目进展记录

## 2026-03-11

### 当前阶段总结

- 任务是 AI6126 Project 1 的人脸语义分割 / face parsing。
- 核心约束是模型可训练参数量必须小于 `1,821,085`。
- 当前主力模型路线是 `deeplab`，不是最早 README 里写的 `attention_unet`。

### 已确认的实验结果

- `weighted_sqrt_d7c3_st20`：最佳 F-score 为 `0.7027261973759753`，出现在第 `26` 个 epoch
- `deeplab_d7c3`：最佳 F-score 为 `0.7022866198657503`，出现在第 `24` 个 epoch
- `deeplab_baseline`：最佳 F-score 为 `0.6937163352614634`，出现在第 `42` 个 epoch

补充说明：

- `best_model.pth` 不是按最后一个 epoch 保存，而是按验证集最高 `F-score` 保存。
- 上述最高分已经通过扫描所有已保存 `metrics.csv` 的历史峰值确认过。
- 当前已保存实验中，最高验证分数是 `0.7027261973759753`，目录为 `checkpoints/deeplab/weighted_sqrt_d7c3_st20/`。



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

- 后续训练应去掉 horizontal flip。

- 从 [src/dataset.py](/Users/chenyixuan/Documents/NTU课程/ACV/AI6126_project1/src/dataset.py) 中移除水平翻转增强。
- 使用当前最优 `deeplab` 配置，在不做水平翻转的前提下重新训练。
- 将新的验证 F-score 与 `weighted_sqrt_d7c3_st20` 做对比。
- 把当前最佳提交候选使用的超参数整理到同一个地方，避免后面写报告时遗漏。
- 在选定的 conda 环境中完整测试 [run_predict_zip.sh](/Users/chenyixuan/Documents/NTU课程/ACV/AI6126_project1/run_predict_zip.sh)。
- 开始整理报告要点，包括：模型选择、loss 设计、数据增强策略，以及为什么移除 horizontal flip。
