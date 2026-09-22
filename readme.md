aloha

# Module 0 学习总结

通过本次作业，我主要学习了以下三个方面：

1. Git 和 GitHub 的基本操作。学会了从 GitHub Classroom clone 远程仓库，在本地使用 `add` 和 `commit` 保存修改，并通过 `push` 将本地提交同步到 GitHub 远程仓库。

2. Git 的分支、合并和版本历史管理。练习了创建并切换 `for_fun` 和 `main` 分支，将一个分支 merge 到另一个分支，并处理 merge conflict。同时，我学会了通过 commit hash checkout 到之前的历史 commit，在 detached HEAD 状态下查看旧版本，再 checkout 回到最新的 commit。这让我更清楚地理解了 Git 的分支和版本历史机制。

3. Python 环境、Hugging Face 和模型推理。我学会了配置 Hugging Face 运行环境，使用预训练的 ResNet 模型在 MNIST 数据集上进行 inference。由于 ResNet 的输入要求，我使用 PyTorch 将 MNIST 的灰度图转换为 RGB，并 resize 到模型需要的尺寸，然后计算模型的 accuracy。这个实验也让我理解了 pretrained model 在未 finetune 的新数据集上可能表现较差。

此外，还了解了代码仓库管理的基本规范，例如使用 `.gitignore`，避免将 dataset 和 model weights 提交到 GitHub。