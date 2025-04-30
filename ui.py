import sys
import os
import json
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QFileDialog, QMessageBox, QGridLayout, QDialog, 
                            QTextEdit, QComboBox, QScrollArea)
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from core import GradeProcessor

class GenerateReportThread(QThread):
    """用于在后台线程中生成AI分析报告"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, processor, num_objectives, current_achievement):
        super().__init__()
        self.processor = processor
        self.num_objectives = num_objectives
        self.current_achievement = current_achievement

    def run(self):
        try:
            self.progress.emit("正在生成AI分析报告...")
            self.processor.generate_ai_report(self.num_objectives, self.current_achievement)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"生成AI分析报告失败：{str(e)}")

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('设置')
        self.setFixedWidth(400)
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 课程简介
        self.description_label = QLabel('课程简介：')
        self.description_input = QTextEdit()
        self.description_input.setFixedHeight(80)
        # 加载已保存的课程简介
        self.description_input.setText(self.parent().course_description)
        layout.addWidget(self.description_label)
        layout.addWidget(self.description_input)
        
        # 课程目标要求（动态）
        self.objectives_layout = QVBoxLayout()
        self.objective_inputs = []
        layout.addLayout(self.objectives_layout)
        
        # 导入上一年达成度表
        self.import_prev_btn = QPushButton('导入上一学年达成度表')
        self.import_prev_btn.clicked.connect(self.import_previous_achievement)
        layout.addWidget(self.import_prev_btn)
        
        # 显示导入文件的路径
        self.file_path_label = QLabel('')
        self.file_path_label.setStyleSheet('font-size: 12px; color: #666666;')
        layout.addWidget(self.file_path_label)
        # 加载已保存的路径
        if self.parent().previous_achievement_file:
            self.file_path_label.setText(self.parent().previous_achievement_file)
        
        # API Key 和检测按钮
        api_layout = QHBoxLayout()
        self.api_key_label = QLabel('API KEY:')
        self.api_key_input = QLineEdit()
        # 加载已保存的 API Key
        self.api_key_input.setText(self.parent().api_key)
        self.test_api_btn = QPushButton('检测')
        self.test_api_btn.clicked.connect(self.test_api_connection)
        api_layout.addWidget(self.api_key_label)
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(self.test_api_btn)
        layout.addLayout(api_layout)
        
        # 保存和清空按钮
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton('保存')
        self.save_btn.clicked.connect(self.save_settings)
        self.clear_btn = QPushButton('清空')
        self.clear_btn.clicked.connect(self.clear_settings)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_objective_inputs(self, num_objectives):
        # 清除现有输入框
        for input_field in self.objective_inputs:
            input_field.setParent(None)
        self.objective_inputs.clear()
        while self.objectives_layout.count():
            item = self.objectives_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 生成新输入框
        parent_objective_requirements = self.parent().objective_requirements
        for i in range(num_objectives):
            label = QLabel(f'课程目标{i+1}要求')
            input_field = QLineEdit()
            input_field.setPlaceholderText(f'请输入目标{i+1}要求')
            # 加载已保存的目标要求
            if i < len(parent_objective_requirements):
                input_field.setText(parent_objective_requirements[i])
            self.objective_inputs.append(input_field)
            self.objectives_layout.addWidget(label)
            self.objectives_layout.addWidget(input_field)
        
        self.adjustSize()  # 自适应高度
    
    def import_previous_achievement(self):
        # 提示用户文件格式要求
        QMessageBox.information(
            self,
            '提示',
            '请确保导入的上一学年达成度表包含以下列：\n'
            '- "课程目标"（如：课程目标1, 课程目标2, ..., 课程总目标）\n'
            '- "上一年度达成度"（达成度数值）\n'
            '或者直接导入程序生成的“课程目标达成度分析表.xlsx”文件。'
        )
        file_name, _ = QFileDialog.getOpenFileName(self, "选择上一学年达成度表", "", "Excel Files (*.xlsx)")
        if file_name:
            self.parent().previous_achievement_file = file_name
            self.file_path_label.setText(file_name)
            QMessageBox.information(self, '成功', f'已选择文件: {os.path.basename(file_name)}')
    
    def save_settings(self):
        """保存设置到配置文件"""
        self.parent().course_description = self.description_input.toPlainText()
        self.parent().objective_requirements = [input_field.text() for input_field in self.objective_inputs]
        self.parent().api_key = self.api_key_input.text()
        self.parent().save_config()
        QMessageBox.information(self, '成功', '设置已保存')
    
    def clear_settings(self):
        """清空设置"""
        self.description_input.clear()
        for input_field in self.objective_inputs:
            input_field.clear()
        self.file_path_label.clear()
        self.parent().course_description = ""
        self.parent().objective_requirements = []
        self.parent().previous_achievement_file = ""
        QMessageBox.information(self, '成功', '设置已清空')
    
    def test_api_connection(self):
        """测试 DeepSeek API 连接"""
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, '警告', '请先输入 API Key')
            return
        
        # 确保 processor 总是被正确初始化
        parent = self.parent()
        if parent.processor is None:
            # 如果 input_file 为空，传入空字符串，因为 test_deepseek_api 不需要 input_file
            input_file = parent.input_file if parent.input_file is not None else ""
            parent.processor = GradeProcessor(
                parent.course_name_input,
                parent.num_objectives_input,
                parent.weight_inputs,
                parent.usual_ratio_input,
                parent.midterm_ratio_input,
                parent.final_ratio_input,
                parent.status_label,
                input_file,
                course_description=parent.course_description,
                objective_requirements=parent.objective_requirements
            )
        
        result = parent.processor.test_deepseek_api(api_key)
        
        if result == "连接成功":
            QMessageBox.information(self, '成功', '链接成功')
        else:
            QMessageBox.critical(self, '失败', f'链接失败：{result}')

class GradeAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.previous_achievement_file = None
        self.course_description = ""
        self.objective_requirements = []
        self.api_key = ""
        self.num_objectives = 0
        self.processor = None
        self.current_achievement = {}
        # 加载保存的配置
        self.load_config()
        self.initUI()
    
    def load_config(self):
        """加载配置文件中的 API Key 和课程设置"""
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
                    self.course_description = config.get('course_description', '')
                    self.objective_requirements = config.get('objective_requirements', [])
                    self.previous_achievement_file = config.get('previous_achievement_file', '')
            except Exception as e:
                print(f"加载配置文件失败: {str(e)}")
    
    def save_config(self):
        """保存 API Key 和课程设置到配置文件"""
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        config = {
            'api_key': self.api_key,
            'course_description': self.course_description,
            'objective_requirements': self.objective_requirements,
            'previous_achievement_file': self.previous_achievement_file
        }
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
    
    def initUI(self):
        # 设置窗口基本属性
        self.setWindowTitle('Scores Calculator')
        self.setMinimumSize(800, 600)  # 设置最小尺寸，允许动态扩展
        self.setStyleSheet("""
            QMainWindow {
                background-color: #E6E6E6;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                padding: 5px;
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                min-height: 25px;
            }
            QTextEdit {
                padding: 5px;
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QPushButton {
                padding: 8px 15px;
                background-color: #808080;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QComboBox {
                padding: 5px;
                background-color: white;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                min-height: 25px;
            }
        """)
        
        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel('课程目标达成度评价计算')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            color: black;
            padding: 10px;
        """)
        layout.addWidget(title_label)
        
        # 添加课程名称输入
        course_layout = QHBoxLayout()
        course_label = QLabel('课程名称:')
        self.course_name_input = QLineEdit()
        course_layout.addWidget(course_label)
        course_layout.addWidget(self.course_name_input)
        course_layout.addStretch()
        layout.addLayout(course_layout)
        
        # 第一行：课程目标数量和权重
        objectives_layout = QVBoxLayout()
        
        # 课程目标数量
        num_label = QLabel('课程目标数量')
        self.num_objectives_input = QLineEdit()
        self.num_objectives_input.setFixedWidth(150)
        
        # 创建水平布局放置数量输入
        num_layout = QHBoxLayout()
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_objectives_input)
        num_layout.addStretch()
        objectives_layout.addLayout(num_layout)
        
        # 权重输入标签
        weights_label = QLabel('课程目标权重系数(总和为1)')
        objectives_layout.addWidget(weights_label)
        
        # 权重输入框容器（使用QGridLayout支持换行）
        self.weights_container = QGridLayout()
        self.weights_container.setSpacing(10)
        self.weights_container.setVerticalSpacing(15)  # 增加行间距，美观
        self.weight_inputs = []
        objectives_layout.addLayout(self.weights_container)
        
        layout.addLayout(objectives_layout)
        
        # 第二行：成绩占比
        ratios_layout = QHBoxLayout()
        ratios_layout.setSpacing(30)  # 输入框之间的间距
        
        # 平时成绩占比
        usual_layout = QVBoxLayout()
        usual_label = QLabel('平时成绩占比')
        self.usual_ratio_input = QLineEdit()
        usual_layout.addWidget(usual_label)
        usual_layout.addWidget(self.usual_ratio_input)
        ratios_layout.addLayout(usual_layout)

        # 添加伸缩空间
        ratios_layout.addStretch()

        # 期中成绩占比
        midterm_layout = QVBoxLayout()
        midterm_label = QLabel('期中成绩占比')
        self.midterm_ratio_input = QLineEdit()
        midterm_layout.addWidget(midterm_label)
        midterm_layout.addWidget(self.midterm_ratio_input)
        ratios_layout.addLayout(midterm_layout)
        
        # 添加伸缩空间
        ratios_layout.addStretch()

        # 期末成绩占比
        final_layout = QVBoxLayout()
        final_label = QLabel('期末成绩占比')
        self.final_ratio_input = QLineEdit()
        final_layout.addWidget(final_label)
        final_layout.addWidget(self.final_ratio_input)
        ratios_layout.addLayout(final_layout)
        
        layout.addLayout(ratios_layout)
        
        # 跨度和分布选择
        mode_layout = QHBoxLayout()
        spread_label = QLabel('分数跨度:')
        self.spread_combo = QComboBox()
        self.spread_combo.addItems(['大跨度 (12-23分)', '中跨度 (7-13分)', '小跨度 (2-8分)'])
        dist_label = QLabel('分布模式:')
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(['正态分布', '左偏态分布', '右偏态分布', '均匀分布'])
        mode_layout.addWidget(spread_label)
        mode_layout.addWidget(self.spread_combo)
        mode_layout.addWidget(dist_label)
        mode_layout.addWidget(self.dist_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # 按钮布局：四个按钮平均分布
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)  # 设置按钮间距
        
        self.import_btn = QPushButton('导入文件')
        self.settings_btn = QPushButton('设置')
        self.export_btn = QPushButton('导出结果')
        self.ai_report_btn = QPushButton('生成AI分析报告')
        
        # 添加按钮到布局
        self.buttons_layout.addWidget(self.import_btn)
        self.buttons_layout.addWidget(self.settings_btn)
        self.buttons_layout.addWidget(self.export_btn)
        self.buttons_layout.addWidget(self.ai_report_btn)
        
        layout.addLayout(self.buttons_layout)
        
        # 添加状态标签
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: #666666;')
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 连接信号
        self.num_objectives_input.textChanged.connect(self.update_weight_inputs)
        self.num_objectives_input.textChanged.connect(self.update_num_objectives)
        self.import_btn.clicked.connect(self.select_file)
        self.settings_btn.clicked.connect(self.open_settings_window)
        self.export_btn.clicked.connect(self.start_analysis)
        self.ai_report_btn.clicked.connect(self.start_generate_ai_report)
        
        # 设置默认值
        self.usual_ratio_input.setText('0.2')
        self.midterm_ratio_input.setText('0.3')
        self.final_ratio_input.setText('0.5')
        
        self.adjust_button_widths()

    def resizeEvent(self, event):
        """重写resizeEvent以实现响应式布局"""
        super().resizeEvent(event)
        self.adjust_button_widths()

    def adjust_button_widths(self):
        """调整按钮宽度以实现平均分布"""
        # 获取主窗口的可用宽度（减去边距）
        window_width = self.width() - 60  # 左右边距共60（30+30）
        button_count = 4  # 四个按钮
        spacing = 10  # 按钮间距
        total_spacing = spacing * (button_count - 1)
        button_width = (window_width - total_spacing) // button_count
        
        # 设置每个按钮的宽度
        self.import_btn.setMinimumWidth(button_width)
        self.settings_btn.setMinimumWidth(button_width)
        self.export_btn.setMinimumWidth(button_width)
        self.ai_report_btn.setMinimumWidth(button_width)

    def update_weight_inputs(self):
        # 清除现有的权重输入框
        for widget in self.weight_inputs:
            widget.setParent(None)
            widget.deleteLater()
        self.weight_inputs.clear()
        
        # 清除布局中的所有项目
        while self.weights_container.count():
            item = self.weights_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        try:
            num_objectives = int(self.num_objectives_input.text())
            if num_objectives <= 0 or num_objectives > 15:  # 限制最大数量
                return
                
            # 使用QGridLayout支持换行，每行最多5个输入框
            columns_per_row = 5
            validator = QDoubleValidator(0.0, 1.0, 2)  # 限制输入0-1的小数
            for i in range(num_objectives):
                weight_input = QLineEdit()
                weight_input.setFixedWidth(80)
                weight_input.setPlaceholderText(f'权重{i+1}')
                weight_input.setValidator(validator)  # 限制输入
                weight_input.textEdited.connect(self.validate_weights_sum)  # 实时校验
                self.weight_inputs.append(weight_input)
                row = i // columns_per_row
                col = i % columns_per_row
                self.weights_container.addWidget(weight_input, row, col)
                
        except ValueError:
            pass

    def validate_weights_sum(self):
        """实时校验权重总和，仅在超过1时提示，并检查未填框"""
        try:
            weights = []
            empty_fields = False
            for input_field in self.weight_inputs:
                text = input_field.text()
                if text:
                    weight = float(text)
                    weights.append(weight)
                else:
                    weights.append(0.0)
                    empty_fields = True
            
            total = sum(weights)
            if total > 1.0:
                QMessageBox.warning(self, '警告', '总权重系数为1')
            
            if abs(total - 1.0) < 0.0001 and empty_fields:
                QMessageBox.warning(self, '警告', '请填写所有权重系数，未填写的请输入0')
        except ValueError:
            pass

    def check_empty_fields(self):
        """检查是否有未填框"""
        empty_fields = False
        for input_field in self.weight_inputs:
            if not input_field.text():
                empty_fields = True
                break
        return empty_fields

    def update_num_objectives(self):
        try:
            self.num_objectives = int(self.num_objectives_input.text())
            if hasattr(self, 'settings_window'):
                self.settings_window.update_objective_inputs(self.num_objectives)
        except ValueError:
            self.num_objectives = 0
    
    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择成绩单文件",
            "",
            "Excel Files (*.xlsx)"
        )
        if file_name:
            self.input_file = file_name
            self.status_label.setText(f"已选择文件: {os.path.basename(file_name)}")
    
    def open_settings_window(self):
        self.settings_window = SettingsWindow(self)
        self.settings_window.update_objective_inputs(self.num_objectives)
        self.settings_window.exec()
        # 保存数据
        self.course_description = self.settings_window.description_input.toPlainText()
        self.objective_requirements = [input_field.text() for input_field in self.settings_window.objective_inputs]
        self.api_key = self.settings_window.api_key_input.text()
        # 保存 API Key 到配置文件
        self.save_config()
    
    def validate_inputs(self):
        try:
            if not self.course_name_input.text():
                raise ValueError("请输入课程名称")
            num_objectives = int(self.num_objectives_input.text())
            weights = []
            for input_field in self.weight_inputs:
                text = input_field.text()
                if not text:
                    raise ValueError("请填写所有权重系数，未填写的请输入0")
                weight = float(text)
                weights.append(weight)
            
            # 仅校验总和是否为1，移除 low <= high 校验
            if abs(sum(weights) - 1) > 0.0001:
                raise ValueError("总权重系数为1")
            
            usual_ratio = float(self.usual_ratio_input.text() or '0.2')
            midterm_ratio = float(self.midterm_ratio_input.text() or '0.3')
            final_ratio = float(self.final_ratio_input.text() or '0.5')
            
            if abs(usual_ratio + midterm_ratio + final_ratio - 1) > 0.0001:
                raise ValueError("成绩占比之和必须等于1")
                
            return num_objectives, weights, usual_ratio, midterm_ratio, final_ratio
            
        except ValueError as e:
            QMessageBox.warning(self, '输入错误', str(e))
            return None

    def start_analysis(self):
        if not self.input_file:
            QMessageBox.warning(self, '错误', '请先选择成绩单文件')
            return
            
        result = self.validate_inputs()
        if result is None:
            return
            
        num_objectives, weights, usual_ratio, midterm_ratio, final_ratio = result
        
        # 获取模式
        spread_mode = {
            '大跨度 (12-23分)': 'large',
            '中跨度 (7-13分)': 'medium',
            '小跨度 (2-8分)': 'small'
        }[self.spread_combo.currentText()]
        distribution = {
            '正态分布': 'normal',
            '左偏态分布': 'left_skewed',
            '右偏态分布': 'right_skewed',
            '均匀分布': 'uniform'
        }[self.dist_combo.currentText()]
        
        self.processor = GradeProcessor(
            self.course_name_input,
            self.num_objectives_input,
            self.weight_inputs,
            self.usual_ratio_input,
            self.midterm_ratio_input,
            self.final_ratio_input,
            self.status_label,
            self.input_file,
            course_description=self.course_description,
            objective_requirements=self.objective_requirements
        )
        
        self.processor.load_previous_achievement(self.previous_achievement_file)
        self.processor.store_api_key(self.api_key)
        
        try:
            self.status_label.setText("正在处理数据...")
            overall_achievement = self.processor.process_grades(num_objectives, weights, usual_ratio, midterm_ratio, final_ratio, 
                                                          spread_mode, distribution)
            # 构造当前达成度数据
            self.current_achievement = {}
            for i in range(1, num_objectives + 1):
                # 从分析报告中提取达成度
                df = pd.read_excel(f"{os.path.dirname(self.input_file)}/{self.course_name_input.text()}课程目标达成度分析表.xlsx")
                m_row = df[df['考核环节'] == '课程分目标达成度\n(M)']
                if not m_row.empty:
                    self.current_achievement[f'课程目标{i}'] = m_row[f'课程目标{i}'].iloc[0] if f'课程目标{i}' in m_row else 0
            self.current_achievement['总达成度'] = overall_achievement
            self.status_label.setText(f"处理完成！课程总目标达成度: {overall_achievement}")
            QMessageBox.information(self, '成功', '分析报告已生成')
        except Exception as e:
            self.status_label.setText("处理失败！")
            QMessageBox.critical(self, '错误', f'处理过程中发生错误：{str(e)}')

    def start_generate_ai_report(self):
        """在后台线程中生成AI分析报告"""
        if self.processor is None:
            QMessageBox.warning(self, '错误', '请先进行成绩分析')
            return
        
        # 创建线程
        self.report_thread = GenerateReportThread(self.processor, self.num_objectives, self.current_achievement)
        self.report_thread.finished.connect(self.on_generate_ai_report_finished)
        self.report_thread.error.connect(self.on_generate_ai_report_error)
        self.report_thread.progress.connect(self.update_status_label)
        self.ai_report_btn.setEnabled(False)  # 禁用按钮，防止重复点击
        self.report_thread.start()

    def on_generate_ai_report_finished(self):
        """生成报告完成时的回调"""
        self.ai_report_btn.setEnabled(True)
        self.status_label.setText("AI分析报告已生成")
        QMessageBox.information(self, '成功', 'AI分析报告已生成')

    def on_generate_ai_report_error(self, error_message):
        """生成报告失败时的回调"""
        self.ai_report_btn.setEnabled(True)
        self.status_label.setText("生成AI分析报告失败！")
        QMessageBox.critical(self, '错误', error_message)

    def update_status_label(self, message):
        """更新状态标签"""
        self.status_label.setText(message)