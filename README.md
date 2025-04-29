课程目标达成度评价计算工具
这是一个基于 PyQt6 的图形界面工具，用于处理学生成绩数据，计算课程目标达成度，并生成详细的分析报告。工具支持导入 Excel 成绩单，设置课程目标权重和成绩占比，导出成绩详情和达成度评价报告。
依赖包
运行本项目需要以下 Python 依赖包：

PyQt6: 用于构建图形用户界面。
pandas: 用于处理 Excel 文件和数据分析。
numpy: 用于数值计算。
openpyxl: 作为 pandas 的 Excel 写入引擎。

安装依赖
为加速下载，推荐使用清华大学的 PyPI 镜像源。执行以下命令安装所有依赖：
pip install PyQt6 pandas numpy openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple

其他国内镜像源（可选）
如果清华镜像源不可用，可尝试以下镜像源：

阿里云：https://mirrors.aliyun.com/pypi/simple/
豆瓣：http://pypi.douban.com/simple/
中国科技大学：https://pypi.mirrors.ustc.edu.cn/simple/

示例（使用阿里云镜像源）：
pip install PyQt6 pandas numpy openpyxl -i https://mirrors.aliyun.com/pypi/simple/

运行项目

确保已安装 Python 3.8 或更高版本。
安装上述依赖包。
下载 grade_calculator3.py 文件。
运行以下命令启动程序：python grade_calculator3.py



使用说明

启动程序后，输入课程名称和课程目标数量。
设置课程目标权重（总和为 1）和成绩占比（平时、期中、期末，总和为 1）。
点击“导入文件”选择 Excel 成绩单文件（需包含学生姓名、平时成绩、期中成绩、期末成绩和总和列）。
点击“导出结果”生成成绩详情和达成度分析报告，结果将保存为 Excel 文件。

注意事项

确保 Python 环境与 PyQt6 兼容。
在虚拟环境中运行时，需先激活虚拟环境。
如遇权限问题，可在命令前添加 sudo（Linux/macOS）或以管理员身份运行命令提示符（Windows）。

