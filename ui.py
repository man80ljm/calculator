import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QFileDialog, QMessageBox, QGridLayout, QDialog, 
                            QTextEdit, QComboBox, QScrollArea)
from PyQt6.QtGui import QFont, QDoubleValidator
from PyQt6.QtCore import Qt
from core import GradeProcessor

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
        
        # API Key
        self.api_key_label = QLabel('API KEY:')
        self.api_key_input = QLineEdit()
        layout.addWidget(self.api_key_label)
        layout.addWidget(self.api_key_input)
        
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
        for i in range(num_objectives):
            label = QLabel(f'课程目标{i+1}要求')
            input_field = QLineEdit()
            input_field.setPlaceholderText(f'请输入目标{i+1}要求')
            self.objective_inputs.append(input_field)
            self.objectives_layout.addWidget(label)
            self.objectives_layout.addWidget(input_field)
        
        self.adjustSize()  # 自适应高度
    
    def import_previous_achievement(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择上一学年达成度表", "", "Excel Files (*.xlsx)")
        if file_name:
            self.parent().previous_achievement_file = file_name
            QMessageBox.information(self, '成功', f'已选择文件: {os.path.basename(file_name)}')

class GradeAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.previous_achievement_file = None
        self.course_description = ""
        self.objective_requirements = []
        self.api_key = ""
        self.num_objectives = 0
        self.initUI()
    
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
        
        # 使用QScrollArea包裹内容
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
        self.ai_report_btn.clicked.connect(self.generate_ai_report)
        
        # 设置默认值
        self.usual_ratio_input.setText('0.2')
        self.midterm_ratio_input.setText('0.3')
        self.final_ratio_input.setText('0.5')
        
        # 初始调整按钮宽度
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
            
            # 如果总和为1，检查是否有未填框
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
            
            # 导出时校验总和是否等于1
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
        
        # 初始化GradeProcessor
        processor = GradeProcessor(
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
        
        # 加载上一年达成度表
        processor.load_previous_achievement(self.previous_achievement_file)
        processor.store_api_key(self.api_key)
        
        try:
            self.status_label.setText("正在处理数据...")
            overall_achievement = processor.process_grades(num_objectives, weights, usual_ratio, midterm_ratio, final_ratio, 
                                                          spread_mode, distribution)
            self.status_label.setText(f"处理完成！课程总目标达成度: {overall_achievement}")
            QMessageBox.information(self, '成功', '分析报告已生成')
        except Exception as e:
            self.status_label.setText("处理失败！")
            QMessageBox.critical(self, '错误', f'处理过程中发生错误：{str(e)}')

    def generate_ai_report(self):
        """生成AI分析报告"""
        processor = GradeProcessor(
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
        processor.store_api_key(self.api_key)
        processor.generate_ai_report()