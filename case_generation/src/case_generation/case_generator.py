import datetime

class CaseGenerator:
    def __init__(self):
        # 定义病例结构模板
        self.case_template = {
            '基本信息': {
                '姓名': '未知',
                '性别': '未知',
                '年龄': '未知',
                '职业': '未知',
                '入院时间': '未知'
            },
            '主诉': '',
            '现病史': '',
            '既往史': '',
            '检查': '',
            '诊断': '',
            '治疗方案': ''
        }
    
    def generate_case(self, text, entities):
        """
        生成结构化病例
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 结构化病例字典
        """
        # 初始化病例
        case = self.case_template.copy()
        
        # 设置入院时间为当前时间
        case['基本信息']['入院时间'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 填充病例各个部分
        case['主诉'] = self._generate_chief_complaint(text, entities)
        case['现病史'] = self._generate_present_illness(text, entities)
        case['既往史'] = self._generate_past_history(text, entities)
        case['检查'] = self._generate_exam(text, entities)
        case['诊断'] = self._generate_diagnosis(entities)
        case['治疗方案'] = self._generate_treatment_plan(text, entities)
        
        return case
    
    def _generate_chief_complaint(self, text, entities):
        """
        生成主诉
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 主诉文本
        """
        # 从文本中提取关键信息作为主诉
        # 首先查找包含症状的部分
        chief_complaint = ""
        
        # 检查文本中是否包含"因"和"入院"
        if "因" in text and "入院" in text:
            start = text.find("因") + 1
            # 查找"入院"之后的内容，直到"既往有"或句子结束
            end = text.find("入院")
            # 查找"既往有"或句子结束
            if "既往有" in text[end:]:
                end = text.find("既往有", end)
            elif "。" in text[end:]:
                end = text.find("。", end)
            else:
                end = len(text)
            
            if start < end:
                # 提取从"因"到"既往有"或句子结束之间的内容
                chief_complaint_part = text[start:end].strip()
                # 移除"需要进行"及后面的内容，因为这属于检查
                if "需要进行" in chief_complaint_part:
                    chief_complaint_part = chief_complaint_part.split("需要进行")[0].strip()
                if chief_complaint_part:
                    chief_complaint = f"患者{chief_complaint_part}"
        
        # 如果没有找到合适的主诉，使用症状信息
        if not chief_complaint and 'SYMPTOM' in entities:
            symptoms = entities['SYMPTOM']
            if symptoms:
                chief_complaint = f"患者{', '.join(symptoms)}{len(symptoms)}天"
        
        # 如果仍然没有主诉，使用文本的前50个字符
        if not chief_complaint:
            chief_complaint = text[:50] + '...' if len(text) > 50 else text
        
        return chief_complaint
    
    def _generate_present_illness(self, text, entities):
        """
        生成现病史
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 现病史文本
        """
        # 构建现病史
        present_illness = f"患者因'{self._generate_chief_complaint(text, entities)}'入院。"
        
        # 添加症状信息
        if 'SYMPTOM' in entities:
            symptoms = entities['SYMPTOM']
            if symptoms:
                present_illness += f" 患者出现{', '.join(symptoms)}等症状。"
        
        # 添加疾病信息
        if 'DISEASE' in entities:
            diseases = entities['DISEASE']
            if diseases:
                present_illness += f" 诊断为{', '.join(diseases)}。"
        
        return present_illness
    
    def _generate_past_history(self, text, entities):
        """
        生成既往史
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 既往史文本
        """
        # 从文本中提取既往史信息
        past_history = ""
        
        # 检查文本中是否包含"既往有"和"病史"
        if "既往有" in text:
            start = text.find("既往有") + 3
            # 查找"病史"或句子结束
            if "病史" in text[start:]:
                end = text.find("病史", start)
                past_history = text[start:end].strip()
            else:
                # 查找下一个逗号或句号
                end_markers = [",", "。"]
                end = len(text)
                for marker in end_markers:
                    if marker in text[start:]:
                        temp_end = text.find(marker, start)
                        if temp_end < end:
                            end = temp_end
                past_history = text[start:end].strip()
        
        # 如果没有从文本中提取到，使用实体信息
        if not past_history and 'DISEASE' in entities:
            diseases = entities['DISEASE']
            if diseases:
                past_history = ', '.join(diseases)
        
        # 如果仍然没有既往史信息，返回"无特殊病史"
        if not past_history:
            past_history = "无特殊病史"
        
        return past_history
    
    def _generate_exam(self, text, entities):
        """
        生成检查信息（合并体格检查和辅助检查）
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 检查文本
        """
        # 从文本中提取检查信息
        exams = []
        
        # 检查文本中是否包含"需要进行"或"进行"
        if "需要进行" in text:
            start = text.find("需要进行") + 4
            # 查找下一个逗号、句号或"，治疗方案包括"
            end_markers = [",", "。", "，治疗方案包括"]
            end = len(text)
            for marker in end_markers:
                if marker in text[start:]:
                    temp_end = text.find(marker, start)
                    if temp_end < end:
                        end = temp_end
            # 提取检查项目
            exam_text = text[start:end].strip()
            # 分割多个检查项目
            if "和" in exam_text:
                exams.extend([item.strip() for item in exam_text.split("和")])
            elif "、" in exam_text:
                exams.extend([item.strip() for item in exam_text.split("、")])
            else:
                exams.append(exam_text)
        elif "进行" in text:
            start = text.find("进行") + 2
            # 查找下一个逗号、句号或"，治疗方案包括"
            end_markers = [",", "。", "，治疗方案包括"]
            end = len(text)
            for marker in end_markers:
                if marker in text[start:]:
                    temp_end = text.find(marker, start)
                    if temp_end < end:
                        end = temp_end
            # 提取检查项目
            exam_text = text[start:end].strip()
            # 分割多个检查项目
            if "和" in exam_text:
                exams.extend([item.strip() for item in exam_text.split("和")])
            elif "、" in exam_text:
                exams.extend([item.strip() for item in exam_text.split("、")])
            else:
                exams.append(exam_text)
        
        # 如果没有从文本中提取到，使用实体信息
        if not exams and 'TEST' in entities:
            tests = entities['TEST']
            if tests:
                exams.extend(tests)
        
        # 如果没有检查信息，返回"无"
        if not exams:
            return "无"
        
        # 合并检查项目
        return ', '.join(exams)
    
    def _generate_diagnosis(self, entities):
        """
        生成诊断
        :param entities: 识别到的医学实体
        :return: 诊断文本
        """
        # 从实体中提取诊断信息
        if 'DISEASE' in entities:
            diseases = entities['DISEASE']
            if diseases:
                return ', '.join(diseases)
        
        return "待明确"
    
    def _generate_treatment_plan(self, text, entities):
        """
        生成治疗方案
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 治疗方案文本
        """
        # 从文本中提取治疗方案信息
        treatment_plan = ""
        
        # 检查文本中是否包含"治疗方案包括"或"使用"
        if "治疗方案包括" in text:
            start = text.find("治疗方案包括") + 6
            # 查找句子结束
            if "。" in text[start:]:
                end = text.find("。", start)
                treatment_plan = text[start:end].strip()
        elif "使用" in text:
            start = text.find("使用") + 2
            # 查找句子结束
            if "。" in text[start:]:
                end = text.find("。", start)
                treatment_plan = "使用" + text[start:end].strip()
        
        # 如果没有从文本中提取到，使用实体信息
        if not treatment_plan:
            treatments = []
            
            # 添加治疗方法信息
            if 'TREATMENT' in entities:
                treatments.extend(entities['TREATMENT'])
            
            # 添加药物信息
            if 'MEDICATION' in entities:
                medications = entities['MEDICATION']
                if medications:
                    if treatments:
                        treatments.append(f"使用{', '.join(medications)}等药物治疗")
                    else:
                        treatments.append(f"使用{', '.join(medications)}等药物治疗")
            
            if treatments:
                treatment_plan = ', '.join(treatments)
        
        # 如果仍然没有治疗方案信息，返回"待明确"
        if not treatment_plan:
            treatment_plan = "待明确"
        
        return treatment_plan
    
    def format_case(self, case):
        """
        格式化病例为可读文本
        :param case: 结构化病例字典
        :return: 格式化后的病例文本
        """
        formatted_case = ""
        
        for section, content in case.items():
            if isinstance(content, dict):
                formatted_case += f"\n{section}:\n"
                for key, value in content.items():
                    formatted_case += f"  {key}: {value}\n"
            else:
                formatted_case += f"\n{section}: {content}\n"
        
        return formatted_case
