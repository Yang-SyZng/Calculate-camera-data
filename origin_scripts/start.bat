@echo off

echo "拉取着色点云。原始图像和CC_xml中"
python scripts/pull_ourdata.py

echo "提取数据，转换为colmap格式"
@REM REM 暂停3秒以看清输出
timeout /t 3 /nobreak
python scripts/convey2ourdata.py

echo "生成深度信息中，此过程耗时较长"
@REM REM 暂停3秒以看清输出
timeout /t 3 /nobreak

python scripts/c2d_new.py
echo "完成，请手动放入作为数据集的降采样点云至xx/sparse/0中，重命名为：points3D.ply"