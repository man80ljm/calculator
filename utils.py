import numpy as np
from typing import List

def normalize_score(score: float) -> float:
    """将分数规范化为整数或整数.5"""
    return round(score * 2) / 2

def get_grade_level(score: float) -> str:
    """返回成绩等级"""
    if score >= 90: return "优秀"
    elif score >= 80: return "良好"
    elif score >= 70: return "中等"
    elif score >= 60: return "合格"
    else: return "不达标"

def calculate_final_score(usual_score: float, midterm_score: float, final_score: float, 
                        usual_ratio: float, midterm_ratio: float, final_ratio: float) -> float:
    """计算最终分数，保留一位小数"""
    return round(usual_score * usual_ratio + midterm_score * midterm_ratio + 
                final_score * final_ratio, 1)

def calculate_achievement_level(grades: List[float]) -> dict:
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

def adjust_column_widths(worksheet) -> None:
    """调整Excel列宽以适应内容"""
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # 获取列字母
        for cell in column:
            try:
                # 计算单元格内容的长度（中文字符按2个字符宽度计算）
                cell_value = str(cell.value)
                length = 0
                for char in cell_value:
                    if ord(char) > 127:  # 假设ASCII大于127的为中文
                        length += 2
                    else:
                        length += 1
                if length > max_length:
                    max_length = length
            except:
                pass
        adjusted_width = max_length + 2  # 额外增加2个单位宽度
        worksheet.column_dimensions[column_letter].width = adjusted_width