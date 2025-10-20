从0到1实现一些基础模型，学习和研究用

mfa通过conda安装 
正确解决方案：使用 Conda 重新安装 MFA
步骤 1：卸载当前 pip 安装的 MFA
# 确保在你的 conda 环境中（stratch-models）
conda activate stratch-models

# 卸载 pip 版本
pip uninstall montreal-forced-aligner

# 从 conda-forge 频道安装（官方推荐）
conda install -c conda-forge montreal-forced-aligner

总结
No module named '_kalpy'
用pip安装 MFA，缺少底层 Kalpy 引擎
改用 conda install -c conda-forge montreal-forced-aligner

