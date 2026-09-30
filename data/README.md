# 数据与样例素材

For the first inference demo, download the Ultralytics bus sample image into this folder:

```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg" -OutFile "data\bus.jpg"
```

下载的样例图片不会提交到 Git，以免未经核对许可就再分发第三方素材。项目代码不会自动下载图片；首次运行前请按根目录 README 的命令操作。

Robot@Home2 是当前室内研究数据集。公开仓库不包含 VOC、Robot@Home2 或派生图片/标注；请从数据集原始提供方获取并遵循其许可。Robot@Home2 原始包和处理后文件应保存在 `data/datasets/RobotAtHome2/`，其中下载包放在 `downloads/`，解压源文件放在 `raw/`，生成数据放在 `processed/`。这些目录由 `.gitignore` 排除。

个人手机视频、CVAT 导出和手工试标文件属于本机私有素材，不随仓库发布，也不用于当前 candidate-v3 LOHO 实验。硬件阶段开始后再恢复个人相机素材测试。
