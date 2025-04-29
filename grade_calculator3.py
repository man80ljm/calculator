import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QFileDialog, QMessageBox, QGridLayout)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import pandas as pd
import numpy as np
from typing import List

class GradeAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_file = None
        
    def initUI(self):
        # 设置窗口基本属性
        self.setWindowTitle('Scores Calculator')
        self.setFixedSize(800, 500)
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
            QPushButton {
                padding: 8px 15px;
                background-color: #808080;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                min-width: 120px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        
        # 创建主窗口部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
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
        
        # 权重输入框容器
        self.weights_container = QHBoxLayout()
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
        
        # 第三行：按钮
        buttons_layout = QHBoxLayout()
        
        self.import_btn = QPushButton('导入文件')
        self.export_btn = QPushButton('导出结果')
        
        buttons_layout.addWidget(self.import_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.export_btn)
        
        layout.addLayout(buttons_layout)
        
        # 添加状态标签
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: #666666;')
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # 连接信号
        self.num_objectives_input.textChanged.connect(self.update_weight_inputs)
        self.import_btn.clicked.connect(self.select_file)
        self.export_btn.clicked.connect(self.start_analysis)
        
        # 设置默认值
        self.usual_ratio_input.setText('0.2')
        self.midterm_ratio_input.setText('0.3')
        self.final_ratio_input.setText('0.5')

    def update_weight_inputs(self):
        # 清除现有的权重输入框
        for widget in self.weight_inputs:
            widget.setParent(None)
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
                
            # 创建新的权重输入框
            for i in range(num_objectives):
                weight_input = QLineEdit()
                weight_input.setFixedWidth(80)
                weight_input.setPlaceholderText(f'权重{i+1}')
                self.weight_inputs.append(weight_input)
                self.weights_container.addWidget(weight_input)
                
        except ValueError:
            pass
            
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
            
    def validate_inputs(self):
        try:
            if not self.course_name_input.text():
                raise ValueError("请输入课程名称")
            num_objectives = int(self.num_objectives_input.text())
            weights = []
            for input_field in self.weight_inputs:
                weight = float(input_field.text() or '0')
                weights.append(weight)
            
            usual_ratio = float(self.usual_ratio_input.text() or '0.2')
            midterm_ratio = float(self.midterm_ratio_input.text() or '0.3')
            final_ratio = float(self.final_ratio_input.text() or '0.5')
            
            if abs(sum(weights) - 1) > 0.0001:
                raise ValueError("权重之和必须等于1")
            
            if abs(usual_ratio + midterm_ratio + final_ratio - 1) > 0.0001:
                raise ValueError("成绩占比之和必须等于1")
                
            return num_objectives, weights, usual_ratio, midterm_ratio, final_ratio
            
        except ValueError as e:
            QMessageBox.warning(self, '输入错误', str(e))
            return None

    def normalize_score(self, score: float) -> float:
        """将分数规范化为整数或整数.5"""
        return round(score * 2) / 2

    def generate_weighted_scores(self, target_sum: float, weights: List[float], min_val: float = 60) -> List[float]:
        """生成在合理范围内波动的加权分数"""
        n = len(weights)
        
        # 修改条件判断方式
        if abs(target_sum) < 0.0001:  # 使用近似值判断是否为0
            return np.zeros(n)
            
        max_attempts = 100
        best_diff = float('inf')
        best_scores = None
        weights_array = np.array(weights)
        
        base_score = target_sum if target_sum >= min_val else min_val
        
        for attempt in range(max_attempts):
            scores = np.zeros(n)
            for i in range(n):
                # 修改权重判断方式
                if abs(weights[i]) > 0.0001:  # 使用近似值判断是否为0
                    score = base_score + np.random.uniform(-5, 5)
                    score = min(99, max(min_val, score))
                    scores[i] = self.normalize_score(score)
                else:
                    scores[i] = 0
            
            weighted_sum = np.sum(scores * weights_array)
            current_diff = abs(weighted_sum - target_sum)
            
            if current_diff < best_diff:
                best_diff = current_diff
                best_scores = scores.copy()
                
            if current_diff < 0.01:
                break
        
        if best_scores is not None:
            scores = best_scores
            while True:
                weighted_sum = np.sum(scores * weights_array)
                if abs(weighted_sum - target_sum) < 0.01:
                    break
                    
                # 修改权重判断方式
                adjustable_indices = [i for i, w in enumerate(weights) if abs(w) > 0.0001]
                if not adjustable_indices:
                    break
                    
                adjust_idx = np.random.choice(adjustable_indices)
                if weighted_sum < target_sum:
                    if scores[adjust_idx] < 99:
                        scores[adjust_idx] += 0.5
                else:
                    if scores[adjust_idx] > min_val:
                        scores[adjust_idx] -= 0.5
        
        return best_scores if best_scores is not None else np.zeros(n)

    def get_grade_level(self, score: float) -> str:
        """返回成绩等级"""
        if score >= 90: return "优秀"
        elif score >= 80: return "良好"
        elif score >= 70: return "中等"
        elif score >= 60: return "合格"
        else: return "不达标"

    def calculate_final_score(self, usual_score: float, midterm_score: float, final_score: float, 
                            usual_ratio: float, midterm_ratio: float, final_ratio: float) -> float:
        """计算最终分数，保留一位小数"""
        return round(usual_score * usual_ratio + midterm_score * midterm_ratio + 
                    final_score * final_ratio, 1)

    def calculate_achievement_level(self, grades: List[float]) -> dict:
        """计算达成度统计"""
        total = len(grades)
        if total == 0:
            return {
                "优秀": 0, "良好": 0, "中等": 0, "合格": 0, "不达标": 0,
                "总人数": 0, "达成度": 0
            }
            
        counts = {
            "优秀": sum(1 for g in grades if g >= 90),
            "良好": sum(1 for g in grades if 80 <= g < 90),
            "中等": sum(1 for g in grades if 70 <= g < 80),
            "合格": sum(1 for g in grades if 60 <= g < 70),
            "不达标": sum(1 for g in grades if g < 60)
        }
        
        # 计算达成度
        achievement = (counts["优秀"]*10 + counts["良好"]*8 + 
                      counts["中等"]*7 + counts["合格"]*6 + 
                      counts["不达标"]*5) / (total * 10)
        
        counts["总人数"] = total
        counts["达成度"] = achievement
        
        return counts

    def generate_achievement_report(self, result_df: pd.DataFrame, weights: List[float], 
                                  course_name: str) -> pd.DataFrame:
        """生成达成度评价报告"""
        objectives = []
        
        # 对每个课程目标进行统计
        for obj_num in range(len(weights)):
            obj_scores = result_df[
                (result_df['课程目标'] != '总和') & 
                (result_df['课程目标'] == obj_num + 1)
            ]['分数'].tolist()
            stats = self.calculate_achievement_level(obj_scores)
            
            objectives.append({
                '课程名称': course_name,
                '课程目标': f'课程目标{obj_num + 1}',
                '优秀': stats['优秀'],
                '良好': stats['良好'],
                '中等': stats['中等'],
                '合格': stats['合格'],
                '不达标': stats['不达标'],
                '学生人数': stats['总人数'],
                '达成度': round(stats['达成度'], 3),
                '课程权重(总100)': weights[obj_num] * 100
            })
        
        # 创建DataFrame
        achievement_df = pd.DataFrame(objectives)
        
        # 计算整体达成度
        overall_achievement = sum(obj['达成度'] * weights[i] 
                                for i, obj in enumerate(objectives))
        
        # 添加整体达成度列
        achievement_df['整体课程目标达成度'] = round(overall_achievement, 3)
        
        return achievement_df

    def generate_objective_analysis_report(self, result_df: pd.DataFrame, course_name: str) -> None:
        """生成课程目标达成度分析报告"""
        output_dir = os.path.dirname(self.input_file)
        analysis_output = os.path.join(output_dir, f'{course_name}课程目标达成度分析表.xlsx')
        
        # 获取课程目标数量
        objectives = sorted([i for i in result_df['课程目标'].unique() if isinstance(i, int)])
        
        # 创建结果数据
        analysis_data = []
        
        # 处理每种考核类型 (平时A、期中B、期末C)
        exam_types = [
            ('平时考核\n(A)', '平时成绩'),
            ('期中考核\n(B)', '期中成绩'),
            ('期末考核\n(C)', '期末成绩')
        ]
        
        # 对每个考核环节计算指标
        for exam_name, score_column in exam_types:
            # 初始化该考核环节的所有目标的平均分
            avg_scores = {}
            score_ratios = {}
            weights = {}
            
            for obj in objectives:
                # 获取当前考核环节的原始分数
                obj_scores = result_df[
                    (result_df['课程目标'] == obj)
                ][score_column].tolist()
                
                if obj_scores:
                    avg_scores[f'课程目标{obj}'] = round(np.mean(obj_scores), 2)
                    score_ratios[f'课程目标{obj}'] = 100
                    weights[f'课程目标{obj}'] = round(
                        result_df[result_df['课程目标'] == obj]['权重'].iloc[0] * 100, 3  # 修改这里
                    )
            
            # 添加数据（每种指标类型只添加一行）
            analysis_data.extend([
                {
                    '考核环节': exam_name,
                    '指标类型': '平均分',
                    **avg_scores
                },
                {
                    '考核环节': exam_name,
                    '指标类型': '分值/满分\n(S)',
                    **score_ratios
                },
                {
                    '考核环节': exam_name,
                    '指标类型': '分权重 (K)',
                    **weights
                }
            ])
        
        # 创建DataFrame
        columns = ['考核环节', '指标类型'] + [f'课程目标{i}' for i in objectives]
        analysis_df = pd.DataFrame(analysis_data, columns=columns)
        
        # 保存到Excel
        with pd.ExcelWriter(analysis_output, engine='openpyxl') as writer:
            analysis_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            # 设置格式
            worksheet = writer.sheets['Sheet1']
            
            # 合并考核环节单元格
            current_exam = None
            start_row = 2
            for i, exam in enumerate(analysis_df['考核环节'], start=2):
                if exam != current_exam:
                    if current_exam is not None and start_row != i:
                        worksheet.merge_cells(f'A{start_row}:A{i-1}')
                    current_exam = exam
                    start_row = i
            
            if start_row != i:
                worksheet.merge_cells(f'A{start_row}:A{i}')
            
            # 调整列宽
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width


    def process_grades(self, num_objectives, weights, usual_ratio, midterm_ratio, final_ratio):
        """处理成绩数据"""
        course_name = self.course_name_input.text()
        if not course_name:
            raise ValueError("请输入课程名称")
            
        output_dir = os.path.dirname(self.input_file)
        detail_output = os.path.join(output_dir, f'{course_name}成绩单详情.xlsx')
        achievement_output = os.path.join(output_dir, f'{course_name}目标达成度评价计算.xlsx')
        
        df = pd.read_excel(self.input_file)
        result_data = []
        
        for idx, row in df.iterrows():
            self.status_label.setText(f"正在处理第 {idx+1}/{len(df)} 个学生的成绩...")
            name = row['学生姓名']
            total_usual = row['平时成绩']
            total_midterm = row['期中成绩']
            total_final = row['期末成绩']
            total_score = row['总和']
            
            min_val = 50 if total_score < 60 else 60
            usual_scores = self.generate_weighted_scores(total_usual, weights, min_val)
            midterm_scores = self.generate_weighted_scores(total_midterm, weights, min_val)
            final_scores = self.generate_weighted_scores(total_final, weights, min_val)
            
            for i in range(num_objectives):
                score = self.calculate_final_score(
                    usual_scores[i], midterm_scores[i], final_scores[i],
                    usual_ratio, midterm_ratio, final_ratio
                )
                
                result_data.append({
                    '学生姓名': name,
                    '课程目标': i + 1,
                    '平时成绩': usual_scores[i],
                    '期中成绩': midterm_scores[i],
                    '期末成绩': final_scores[i],
                    '权重': weights[i],
                    '平时成绩占比': usual_ratio,
                    '期中成绩占比': midterm_ratio,
                    '期末成绩占比': final_ratio,
                    '分数': score,
                    '等级': self.get_grade_level(score)
                })
            
            final_total_score = self.calculate_final_score(
                total_usual, total_midterm, total_final,
                usual_ratio, midterm_ratio, final_ratio
            )
            
            result_data.append({
                '学生姓名': name,
                '课程目标': '总和',
                '平时成绩': total_usual,
                '期中成绩': total_midterm,
                '期末成绩': total_final,
                '权重': sum(weights),
                '平时成绩占比': usual_ratio,
                '期中成绩占比': midterm_ratio,
                '期末成绩占比': final_ratio,
                '分数': final_total_score,
                '等级': self.get_grade_level(final_total_score)
            })
        
        # 创建结果DataFrame并保存
        result_df = pd.DataFrame(result_data)
        
        with pd.ExcelWriter(detail_output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            worksheet = writer.sheets['Sheet1']
            current_name = None
            start_row = 2
            
            for i, name in enumerate(result_df['学生姓名'], start=2):
                if name != current_name:
                    if current_name is not None and start_row != i-1:
                        worksheet.merge_cells(f'A{start_row}:A{i-1}')
                    current_name = name
                    start_row = i
            
            if start_row != i:
                worksheet.merge_cells(f'A{start_row}:A{i}')
                
            # 调整列宽以适应内容
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # 生成并保存达成度评价报告
        achievement_df = self.generate_achievement_report(result_df, weights, course_name)
        
        with pd.ExcelWriter(achievement_output, engine='openpyxl') as writer:
            achievement_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            worksheet = writer.sheets['Sheet1']
            course_name_col = 'A'
            start_row = 2
            end_row = len(achievement_df) + 1
            
            if end_row > start_row:
                cell_range = f'{course_name_col}{start_row}:{course_name_col}{end_row}'
                worksheet.merge_cells(cell_range)
                
            column_index = list(achievement_df.columns).index('整体课程目标达成度') + 1
            column_letter = worksheet.cell(row=1, column=column_index).column_letter
        
            worksheet.merge_cells(f"{column_letter}{start_row}:{column_letter}{end_row}")
            overall_achievement = achievement_df['整体课程目标达成度'].iloc[0]
            worksheet[f"{column_letter}{start_row}"] = overall_achievement
                
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        # 在现有的两个报告生成之后添加
        self.generate_objective_analysis_report(result_df, course_name)

    def start_analysis(self):
        if not self.input_file:
            QMessageBox.warning(self, '错误', '请先选择成绩单文件')
            return
            
        result = self.validate_inputs()
        if result is None:
            return
            
        num_objectives, weights, usual_ratio, midterm_ratio, final_ratio = result
        
        try:
            self.status_label.setText("正在处理数据...")
            self.process_grades(num_objectives, weights, usual_ratio, midterm_ratio, final_ratio)
            self.status_label.setText("处理完成！")
            QMessageBox.information(self, '成功', '分析报告已生成')
        except Exception as e:
            self.status_label.setText("处理失败！")
            QMessageBox.critical(self, '错误', f'处理过程中发生错误：{str(e)}')

def main():
    app = QApplication(sys.argv)
    ex = GradeAnalysisApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()