import os
import numpy as np
import pandas as pd
from typing import List
from utils import normalize_score, get_grade_level, calculate_final_score, calculate_achievement_level, adjust_column_widths

class GradeProcessor:
    def __init__(self, course_name_input, num_objectives_input, weight_inputs, usual_ratio_input, 
                 midterm_ratio_input, final_ratio_input, status_label, input_file, 
                 course_description="", objective_requirements=None):
        self.course_name_input = course_name_input
        self.num_objectives_input = num_objectives_input
        self.weight_inputs = weight_inputs
        self.usual_ratio_input = usual_ratio_input
        self.midterm_ratio_input = midterm_ratio_input
        self.final_ratio_input = final_ratio_input
        self.status_label = status_label
        self.input_file = input_file
        self.course_description = course_description
        self.objective_requirements = objective_requirements or []
        self.previous_achievement_data = None
        self.api_key = None

    def calculate_score_bounds(self, target_score: float, spread_mode: str) -> tuple:
        """
        根据目标分数和跨度模式计算分数的上下限，动态调整低分情况的跨度。
        """
        spread_ranges = {'large': 23, 'medium': 13, 'small': 8}
        base_spread = spread_ranges[spread_mode]

        if target_score < 40:
            spread = min(base_spread, target_score + 5)
        else:
            spread = base_spread

        min_bound = max(0.0, target_score - spread)
        max_bound = min(99.0, target_score + spread)
        
        return min_bound, max_bound

    def generate_weighted_scores(self, target_sum: float, weights: List[float], 
                                all_scores: List[List[float]], spread_mode: str = 'medium', 
                                distribution: str = 'uniform') -> List[float]:
        """
        根据目标总分和权重反推出各课程目标的分数，同时考虑分布和跨度。
        """
        n = len(weights)
        if abs(target_sum) < 0.0001:
            return np.zeros(n).tolist()

        # 验证权重总和
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.0001:
            raise ValueError(f"Weights sum must be 1, got {weight_sum}")

        # 计算分数的上下限
        min_bound, max_bound = self.calculate_score_bounds(target_sum, spread_mode)

        # 计算所有学生的成绩分布参数
        all_scores_flat = [score for student_scores in all_scores for score in student_scores]
        if all_scores_flat:
            mean_score = np.mean(all_scores_flat)
            std_score = np.std(all_scores_flat) if len(all_scores_flat) > 1 else 10.0
        else:
            mean_score = target_sum
            std_score = 10.0

        # 计算基准分数
        base_score = target_sum / weight_sum
        base_score = max(min_bound, min(max_bound, base_score))

        # 动态选择分布模式：低分时默认右偏分布
        effective_distribution = distribution
        if target_sum < 40 and distribution == 'normal':
            effective_distribution = 'right_skewed'

        # 根据分布模式和跨度模式生成初始分数
        scores = np.zeros(n)
        # 定义 scale，确保所有分布模式都有一个 scale 值
        if spread_mode == 'large':
            scale = min(std_score, (max_bound - min_bound) / 3)  # 增大波动
        elif spread_mode == 'medium':
            scale = min(std_score, (max_bound - min_bound) / 4)
        else:  # small
            scale = min(std_score, (max_bound - min_bound) / 5)

        if effective_distribution == 'normal':
            scores = np.random.normal(base_score, scale, n)
        elif effective_distribution == 'left_skewed':
            scores = np.random.beta(5, 2, n) * (max_bound - min_bound) + min_bound
            # 强制调整左偏分布：将部分分数向上偏移
            scores += np.random.uniform(2, 5, n)
        elif effective_distribution == 'right_skewed':
            scores = np.random.beta(2, 5, n) * (max_bound - min_bound) + min_bound
            # 强制调整右偏分布：将部分分数向下偏移
            scores -= np.random.uniform(2, 5, n)
        else:  # uniform
            scores = np.random.uniform(min_bound, max_bound, n)

        scores = np.clip(scores, min_bound, max_bound)
        scores = np.round(scores, 1)

        # 归一化初始分数
        weights_array = np.array(weights)
        current_sum = np.sum(scores * weights_array)
        if abs(current_sum - target_sum) > 0.1:
            scale_factor = target_sum / current_sum if current_sum != 0 else 1.0
            scores = scores * scale_factor
            scores = np.clip(scores, min_bound, max_bound)
            scores = np.round(scores, 1)

        # 添加随机扰动，增强分布多样性
        scores += np.random.normal(0, scale, n)
        scores = np.clip(scores, min_bound, max_bound)
        scores = np.round(scores, 1)

        # 调试日志
        print(f"Initial scores (after scaling and perturbation): {scores.tolist()}")
        print(f"Initial weighted sum: {np.sum(scores * weights_array)}, Target sum: {target_sum}")

        # 微调分数以精确匹配目标加权和
        max_attempts = 1000
        for attempt in range(max_attempts):
            current_sum = np.sum(scores * weights_array)
            diff = target_sum - current_sum
            if abs(diff) < 0.1:
                break

            # 优先调整偏离均值较大的分数
            mean_score = np.mean(scores)
            deviations = [abs(score - mean_score) for score in scores]
            adjustable_indices = [i for i, w in enumerate(weights) if w > 0.0001]
            if not adjustable_indices:
                break

            # 按偏差大小选择调整的分数
            probabilities = [deviations[i] / sum(deviations[i] for i in adjustable_indices) for i in adjustable_indices]
            idx = np.random.choice(adjustable_indices, p=probabilities)

            weight = weights[idx]
            adjustment = diff / weight
            step = 0.5 if abs(adjustment) >= 0.5 else (0.1 if abs(adjustment) >= 0.1 else 0.01)
            adjustment = step if diff > 0 else -step

            # 模拟退火：增加随机性
            if np.random.rand() < 0.3:
                adjustment = np.random.choice([-2.0, -1.0, 1.0, 2.0])

            scores[idx] += adjustment
            scores[idx] = max(min(scores[idx], max_bound), min_bound)
            scores = np.round(scores, 1)

            # 调试日志
            if attempt % 100 == 0:
                print(f"Attempt {attempt}, Current sum: {np.sum(scores * weights_array)}, Diff: {diff}")

        final_sum = np.sum(scores * weights_array)
        if abs(final_sum - target_sum) > 0.1:
            raise ValueError(f"Generated scores do not match target sum: {final_sum} != {target_sum}")

        # 验证分布多样性
        final_std = np.std(scores)
        if final_std < 6.0:
            scores += np.random.normal(0, scale, n)
            scores = np.clip(scores, min_bound, max_bound)
            scores = np.round(scores, 1)

        # 最终调试日志
        print(f"Final scores: {scores.tolist()}")
        print(f"Final weighted sum: {final_sum}")
        print(f"Final distribution - Mean: {np.mean(scores):.2f}, Std: {final_std:.2f}")

        return scores.tolist()

    def process_grades(self, num_objectives, weights, usual_ratio, midterm_ratio, final_ratio, 
                      spread_mode='medium', distribution='uniform'):
        """处理成绩数据"""
        course_name = self.course_name_input.text()
        if not course_name:
            raise ValueError("请输入课程名称")
            
        output_dir = os.path.dirname(self.input_file)
        detail_output = os.path.join(output_dir, f'{course_name}成绩单详情.xlsx')
        # achievement_output = os.path.join(output_dir, f'{course_name}目标达成度评价计算.xlsx')  # 注释掉，不再导出

        # 读取 Excel 文件并校验列名
        df = pd.read_excel(self.input_file)
        required_columns = ['学生姓名', '平时成绩', '期中成绩', '期末成绩', '总和']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入文件缺少以下必需列：{', '.join(missing_columns)}。请确保文件包含：{', '.join(required_columns)}")
        
        result_data = []
        all_usual_scores = [[] for _ in range(num_objectives)]
        all_midterm_scores = [[] for _ in range(num_objectives)]
        all_final_scores = [[] for _ in range(num_objectives)]
        
        for idx, row in df.iterrows():
            self.status_label.setText(f"正在处理第 {idx+1}/{len(df)} 个学生的成绩...")
            name = row['学生姓名']
            total_usual = row['平时成绩']
            total_midterm = row['期中成绩']
            total_final = row['期末成绩']
            total_score = row['总和']
            
            usual_scores = self.generate_weighted_scores(total_usual, weights, all_usual_scores, spread_mode, distribution)
            midterm_scores = self.generate_weighted_scores(total_midterm, weights, all_midterm_scores, spread_mode, distribution)
            final_scores = self.generate_weighted_scores(total_final, weights, all_final_scores, spread_mode, distribution)
            
            for i in range(num_objectives):
                all_usual_scores[i].append(usual_scores[i])
                all_midterm_scores[i].append(midterm_scores[i])
                all_final_scores[i].append(final_scores[i])
            
            for i in range(num_objectives):
                score = calculate_final_score(
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
                    '等级': get_grade_level(score)
                })
            
            final_total_score = calculate_final_score(
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
                '等级': get_grade_level(final_total_score)
            })
        
        result_df = pd.DataFrame(result_data)
        
        with pd.ExcelWriter(detail_output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            worksheet = writer.sheets['Sheet1']
            if self.course_description:
                worksheet.cell(row=1, column=1).value = f"课程简介: {self.course_description}"
            
            current_name = None
            start_row = 2 if not self.course_description else 3
            
            for i, name in enumerate(result_df['学生姓名'], start=start_row):
                if name != current_name:
                    if current_name is not None and start_row != i-1:
                        worksheet.merge_cells(f'A{start_row}:A{i-1}')
                    current_name = name
                    start_row = i
            
            if start_row != i:
                worksheet.merge_cells(f'A{start_row}:A{i}')
                
            adjust_column_widths(worksheet)
        
        overall_achievement = self.generate_objective_analysis_report(result_df, course_name, weights, usual_ratio, midterm_ratio, final_ratio)
        return overall_achievement

    def generate_achievement_report(self, result_df: pd.DataFrame, weights: List[float], 
                                  course_name: str) -> pd.DataFrame:
        """生成达成度评价报告（暂时保留方法，但不调用）"""
        objectives = []
        
        for obj_num in range(len(weights)):
            obj_scores = result_df[
                (result_df['课程目标'] != '总和') & 
                (result_df['课程目标'] == obj_num + 1)
            ]['分数'].tolist()
            stats = calculate_achievement_level(obj_scores)
            
            objective_data = {
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
            }
            
            if self.previous_achievement_data is not None:
                prev_achievement = self.previous_achievement_data[
                    self.previous_achievement_data['课程目标'] == f'课程目标{obj_num + 1}'
                ]
                if not prev_achievement.empty:
                    objective_data['上一学年达成度'] = round(float(prev_achievement['达成度'].iloc[0]), 3)
                else:
                    objective_data['上一学年达成度'] = None
            
            objectives.append(objective_data)
        
        achievement_df = pd.DataFrame(objectives)
        
        overall_achievement = sum(obj['达成度'] * weights[i] 
                                for i, obj in enumerate(objectives))
        
        achievement_df['整体课程目标达成度'] = round(overall_achievement, 3)
        
        if self.previous_achievement_data is not None:
            prev_overall = self.previous_achievement_data['整体课程目标达成度'].iloc[0] if '整体课程目标达成度' in self.previous_achievement_data.columns else None
            achievement_df['上一学年整体达成度'] = round(float(prev_overall), 3) if prev_overall is not None else None
        
        return achievement_df

    def generate_objective_analysis_report(self, result_df: pd.DataFrame, course_name: str, weights, usual_ratio, midterm_ratio, final_ratio) -> float:
        """生成课程目标达成度分析报告"""
        output_dir = os.path.dirname(self.input_file)
        analysis_output = os.path.join(output_dir, f'{course_name}课程目标达成度分析表.xlsx')
        
        objectives = sorted([i for i in result_df['课程目标'].unique() if isinstance(i, int)])
        
        analysis_data = []
        
        exam_types = [
            ('平时考核\n(A)', '平时成绩'),
            ('期中考核\n(B)', '期中成绩'),
            ('期末考核\n(C)', '期末成绩')
        ]
        
        weights_dict = {f'课程目标{obj}': round(w * 100, 3) for obj, w in zip(objectives, weights)}
        
        m_values = {}
        
        for exam_name, score_column in exam_types:
            avg_scores = {}
            score_ratios = {}
            
            for obj in objectives:
                obj_scores = result_df[
                    (result_df['课程目标'] == obj)
                ][score_column].tolist()
                
                if obj_scores:
                    avg_scores[f'课程目标{obj}'] = round(np.mean(obj_scores), 1)
                    score_ratios[f'课程目标{obj}'] = 100
            
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
                    **weights_dict
                }
            ])
        
        m_row = {'考核环节': '课程分目标达成度\n(M)', '指标类型': ''}
        for obj in objectives:
            usual_avg = analysis_data[0].get(f'课程目标{obj}', 0)
            midterm_avg = analysis_data[3].get(f'课程目标{obj}', 0)
            final_avg = analysis_data[6].get(f'课程目标{obj}', 0)
            m = usual_avg * usual_ratio + midterm_avg * midterm_ratio + final_avg * final_ratio
            m_row[f'课程目标{obj}'] = round(m, 1)
            m_values[obj] = m
        
        analysis_data.append(m_row)
        
        z_row = {'考核环节': '课程分目标总权重\n(Z)', '指标类型': ''}
        for obj in objectives:
            z_row[f'课程目标{obj}'] = weights_dict[f'课程目标{obj}']
        analysis_data.append(z_row)
        
        total_achievement = sum(m_values[obj] * weights[obj-1] for obj in objectives)
        total_achievement = round(total_achievement, 1)
        total_row = {'考核环节': '课程总目标达成度', '指标类型': ''}
        for obj in objectives:
            total_row[f'课程目标{obj}'] = total_achievement if obj == objectives[0] else ''
        analysis_data.append(total_row)
        
        columns = ['考核环节', '指标类型'] + [f'课程目标{i}' for i in objectives]
        analysis_df = pd.DataFrame(analysis_data, columns=columns)
        
        with pd.ExcelWriter(analysis_output, engine='openpyxl') as writer:
            analysis_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            worksheet = writer.sheets['Sheet1']
            
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
            
            m_row_idx = len(analysis_df) - 2 + 1
            worksheet.merge_cells(f'A{m_row_idx}:B{m_row_idx}')
            
            z_row_idx = len(analysis_df) - 1 + 1
            worksheet.merge_cells(f'A{z_row_idx}:B{z_row_idx}')
            
            total_row_idx = len(analysis_df) + 1
            worksheet.merge_cells(f'A{total_row_idx}:B{total_row_idx}')
            worksheet.merge_cells(f'C{total_row_idx}:{chr(ord("C") + len(objectives) - 1)}{total_row_idx}')
            
            adjust_column_widths(worksheet)
        
        return total_achievement

    def load_previous_achievement(self, file_path: str) -> None:
        """加载上一学年达成度表"""
        if not file_path:
            return
        try:
            self.previous_achievement_data = pd.read_excel(file_path)
            if self.status_label:
                self.status_label.setText(f"已加载上一学年达成度表: {os.path.basename(file_path)}")
        except Exception as e:
            if self.status_label:
                self.status_label.setText("加载上一学年达成度表失败！")
            raise ValueError(f"加载上一学年达成度表失败: {str(e)}")

    def store_api_key(self, api_key: str) -> None:
        """存储API Key"""
        self.api_key = api_key
        if self.status_label:
            self.status_label.setText("已存储API Key")

    def generate_ai_report(self) -> None:
        """生成AI分析报告（占位）"""
        if not self.api_key:
            if self.status_label:
                self.status_label.setText("请先设置API Key")
            return
        if self.status_label:
            self.status_label.setText(f"API Key: {self.api_key}，AI分析报告功能待实现")