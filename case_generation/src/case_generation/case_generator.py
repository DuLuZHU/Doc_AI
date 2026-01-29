import datetime
import requests
import json

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
            '体格检查': '',
            '辅助检查': '',
            '诊断': '',
            '处理意见': ''
        }
    
    def generate_case(self, text, entities):
        """
        生成结构化病例
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 结构化病例字典
        """
        # 判断文本复杂度，如果文本较长或包含多个信息点，使用LLM
        if len(text) > 100 or ('，' in text and text.count('，') > 3):
            try:
                # 尝试使用LLM生成病例
                case = self._generate_case_with_llm(text)
                if case:
                    return case
            except Exception as e:
                # 如果LLM调用失败，回退到基于规则的方法
                print(f"LLM调用失败，回退到基于规则的方法: {e}")
        
        # 初始化病例
        case = self.case_template.copy()
        
        # 设置入院时间为当前时间
        case['基本信息']['入院时间'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 提取基本信息
        case['基本信息'] = self._extract_basic_info(text, case['基本信息'])
        
        # 填充病例各个部分
        case['主诉'] = self._generate_chief_complaint(text, entities)
        case['现病史'] = self._generate_present_illness(text, entities)
        case['既往史'] = self._generate_past_history(text, entities)
        case['体格检查'] = self._generate_physical_exam(text, entities)
        case['辅助检查'] = self._generate_auxiliary_exam(text, entities)
        case['诊断'] = self._generate_diagnosis(text, entities)
        case['处理意见'] = self._generate_disposal_opinion(text, entities)
        
        return case
    
    def _generate_case_with_llm(self, text):
        """
        使用LLM生成结构化病例
        :param text: 输入文本
        :return: 结构化病例字典
        """
        try:
            # 构建提示词
            prompt = f"""请根据以下文本生成结构化病例：
{text}

病例结构应包括：
1. 基本信息（姓名、性别、年龄、职业、入院时间）
2. 主诉（简明扼要的症状描述）
3. 现病史（具体症状）
4. 既往史（过去的疾病史）
5. 体格检查（生命体征等）
6. 辅助检查（检验、检查结果）
7. 诊断（当前诊断的疾病）
8. 处理意见（检查项目、治疗措施、药物等）

请严格按照上述结构输出，每项内容要准确反映文本中的信息。

输出格式应为JSON，示例：
{{
  "基本信息": {{
    "姓名": "张三",
    "性别": "女",
    "年龄": "45",
    "职业": "未知",
    "入院时间": "2026-01-28 12:00:00"
  }},
  "主诉": "腹痛、腹泻3天，伴有发热",
  "现病史": "腹痛、腹泻3天，伴有发热，体温38.5℃",
  "既往史": "溃疡性结肠炎",
  "体格检查": "体温：38.5℃",
  "辅助检查": "白细胞数：10*10^20，白细胞异常有发炎症状",
  "诊断": "溃疡性结肠炎",
  "处理意见": "结肠镜检查和粪便常规检查，使用抗生素和糖皮质激素，头孢每次100mg，每日2次"
}}
"""
            
            # 调用自定义API
            
            # 设置API参数
            api_url = "https://www.deepseek.cn/v1/chat/completions"
            api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            model = "deepseek-reasoner"
            
            # 构建请求体
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的医学病例生成助手，能够根据患者描述生成结构化的医学病例。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            print(f"开始调用LLM API: {api_url}")
            print(f"使用模型: {model}")
            print(f"请求体长度: {len(str(payload))} 字符")
            
            # 发送请求
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            print(f"API响应状态码: {response.status_code}")
            print(f"API响应头: {dict(response.headers)}")
            print(f"API响应内容: {response.text}")
            
            try:
                response.raise_for_status()  # 检查请求是否成功
            except requests.exceptions.HTTPError as e:
                print(f"HTTP错误详情: {e}")
                raise
            
            # 解析响应
            response_data = response.json()
            print(f"API响应数据: {response_data}")
            
            case_json = response_data['choices'][0]['message']['content']
            print(f"LLM生成的病例JSON: {case_json}")
            
            case = json.loads(case_json)
            print(f"解析后的病例: {case}")
            
            return case
        except ImportError as e:
            # 如果模块不可用，返回None
            print(f"导入模块失败: {e}")
            return None
        except requests.exceptions.RequestException as e:
            # 处理请求异常
            print(f"API请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            # 处理JSON解析异常
            print(f"JSON解析失败: {e}")
            return None
        except KeyError as e:
            # 处理键错误
            print(f"响应数据结构错误: {e}")
            return None
        except Exception as e:
            # 处理其他异常
            print(f"其他错误: {e}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return None
    
    def _extract_basic_info(self, text, basic_info):
        """
        从文本中提取基本信息
        :param text: 输入文本
        :param basic_info: 基本信息字典
        :return: 更新后的基本信息字典
        """
        # 提取年龄
        import re
        age_match = re.search(r'(\d+)岁', text)
        if age_match:
            basic_info['年龄'] = age_match.group(1)
        
        # 提取性别
        if '男性' in text or '男' in text:
            basic_info['性别'] = '男'
        elif '女性' in text or '女' in text:
            basic_info['性别'] = '女'
        
        # 提取姓名
        name_match = re.search(r'患者姓名([^，]+)', text)
        if not name_match:
            name_match = re.search(r'患者([^，]+)，', text)
        if name_match:
            name = name_match.group(1).strip()
            # 过滤掉性别、年龄等信息，只保留姓名
            if '男' not in name and '女' not in name and '岁' not in name:
                basic_info['姓名'] = name
        
        return basic_info
    
    def _generate_chief_complaint(self, text, entities):
        """
        生成主诉
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 主诉文本
        """
        # 从文本中提取关键信息作为主诉
        chief_complaint = ""
        
        # 检查文本中是否包含"因"和"入院"
        if "因" in text and "入院" in text:
            # 提取整句话作为主诉
            if "。" in text:
                end = text.find("。")
                chief_complaint = text[:end].strip()
            else:
                chief_complaint = text.strip()
        
        # 如果没有找到合适的主诉，使用症状信息
        if not chief_complaint and 'SYMPTOM' in entities:
            symptoms = entities['SYMPTOM']
            if symptoms:
                chief_complaint = f"患者{', '.join(symptoms)}{len(symptoms)}天入院"
        
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
        # 从文本中提取现病史信息
        present_illness = ""
        
        # 检查文本中是否包含症状
        if 'SYMPTOM' in entities:
            symptoms = entities['SYMPTOM']
            if symptoms:
                present_illness = f"{', '.join(symptoms)}等症状"
        
        # 检查文本中是否包含疾病
        elif 'DISEASE' in entities:
            diseases = entities['DISEASE']
            if diseases:
                present_illness = ', '.join(diseases)
        
        # 如果文本中包含具体症状描述，直接提取
        elif "因" in text and "入院" in text:
            start = text.find("因") + 1
            end = text.find("入院")
            if start < end:
                present_illness = text[start:end].strip()
                # 移除"伴有"后面的内容
                if "伴有" in present_illness:
                    present_illness = present_illness.split("伴有")[0].strip()
        
        # 如果仍然没有现病史信息，返回"无"
        if not present_illness:
            present_illness = "无"
        
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
        
        # 处理特殊情况：高血压就诊
        elif "高血压" in text and "就诊" in text:
            past_history = "高血压"
        
        # 如果仍然没有既往史信息，返回"无特殊病史"
        if not past_history:
            past_history = "无特殊病史"
        
        return past_history
    
    def _generate_physical_exam(self, text, entities):
        """
        生成体格检查信息
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 体格检查文本
        """
        # 从文本中提取体格检查信息
        physical_exam = []
        
        # 提取体温信息
        import re
        temperature_match = re.search(r'体温([\d.]+℃)', text)
        if temperature_match:
            physical_exam.append(f"体温：{temperature_match.group(1)}")
        
        # 提取脉搏信息
        pulse_match = re.search(r'脉搏([\d.]+次/分)', text)
        if pulse_match:
            physical_exam.append(f"脉搏：{pulse_match.group(1)}")
        
        # 提取呼吸信息
        respiration_match = re.search(r'呼吸([\d.]+次/分)', text)
        if respiration_match:
            physical_exam.append(f"呼吸：{respiration_match.group(1)}")
        
        # 提取血压信息
        blood_pressure_match = re.search(r'血压([\d./]+mmHg)', text)
        if blood_pressure_match:
            physical_exam.append(f"血压：{blood_pressure_match.group(1)}")
        
        # 如果没有提取到任何信息，返回"无"
        if not physical_exam:
            return "无"
        
        # 合并体格检查项目
        return ', '.join(physical_exam)
    
    def _generate_auxiliary_exam(self, text, entities):
        """
        生成辅助检查信息
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 辅助检查文本
        """
        # 从文本中提取辅助检查信息
        auxiliary_exam = []
        
        # 检查文本中是否包含检验、检查结果
        # 提取白细胞数
        import re
        white_blood_cell_match = re.search(r'白细胞数[为：:]([\d.]+[*×]10[^，。]+)', text)
        if not white_blood_cell_match:
            white_blood_cell_match = re.search(r'白细胞([\d.]+[*×]10[^，。]+)', text)
        if white_blood_cell_match:
            auxiliary_exam.append(f"白细胞数：{white_blood_cell_match.group(1)}")
        
        # 提取红细胞数
        red_blood_cell_match = re.search(r'红细胞数[为：:]([\d.]+[*×]10[^，。]+)', text)
        if not red_blood_cell_match:
            red_blood_cell_match = re.search(r'红细胞([\d.]+[*×]10[^，。]+)', text)
        if red_blood_cell_match:
            auxiliary_exam.append(f"红细胞数：{red_blood_cell_match.group(1)}")
        
        # 提取血红蛋白
        hemoglobin_match = re.search(r'血红蛋白[为：:]([\d.]+g/L)', text)
        if not hemoglobin_match:
            hemoglobin_match = re.search(r'血红蛋白([\d.]+g/L)', text)
        if hemoglobin_match:
            auxiliary_exam.append(f"血红蛋白：{hemoglobin_match.group(1)}")
        
        # 提取血小板数
        platelet_match = re.search(r'血小板数[为：:]([\d.]+[*×]10[^，。]+)', text)
        if not platelet_match:
            platelet_match = re.search(r'血小板([\d.]+[*×]10[^，。]+)', text)
        if platelet_match:
            auxiliary_exam.append(f"血小板数：{platelet_match.group(1)}")
        
        # 提取其他检验结果
        other_test_match = re.search(r'([^，。]+[\d.]+[^，。]+异常)', text)
        if not other_test_match:
            other_test_match = re.search(r'报告显示([^，。]+)', text)
        if other_test_match:
            auxiliary_exam.append(other_test_match.group(1))
        
        # 如果没有提取到任何信息，返回"无"
        if not auxiliary_exam:
            return "无"
        
        # 合并辅助检查项目
        return ', '.join(auxiliary_exam)
    
    def _generate_diagnosis(self, text, entities):
        """
        生成诊断
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 诊断文本
        """
        # 从文本中提取诊断信息
        diagnosis = ""
        
        # 检查文本中是否包含"诊断为"
        if "诊断为" in text:
            start = text.find("诊断为") + 3
            # 查找下一个逗号或句号
            end_markers = [",", "。"]
            end = len(text)
            for marker in end_markers:
                if marker in text[start:]:
                    temp_end = text.find(marker, start)
                    if temp_end < end:
                        end = temp_end
            diagnosis = text[start:end].strip()
        
        # 处理特殊情况：高血压就诊
        elif "高血压" in text and "就诊" in text:
            diagnosis = "高血压"
        
        # 处理特殊情况：溃疡性结肠炎
        elif "溃疡性结肠炎" in text:
            diagnosis = "溃疡性结肠炎"
        
        # 如果没有从文本中提取到，使用实体信息
        if not diagnosis and 'DISEASE' in entities:
            diseases = entities['DISEASE']
            if diseases:
                # 过滤掉既往史中的疾病
                filtered_diseases = []
                for disease in diseases:
                    if "既往有" not in text or disease not in text.split("既往有")[1]:
                        filtered_diseases.append(disease)
                if filtered_diseases:
                    diagnosis = ', '.join(filtered_diseases)
        
        # 如果仍然没有诊断信息，返回"待明确"
        if not diagnosis:
            diagnosis = "待明确"
        
        return diagnosis
    
    def _generate_disposal_opinion(self, text, entities):
        """
        生成处理意见
        :param text: 输入文本
        :param entities: 识别到的医学实体
        :return: 处理意见文本
        """
        # 从文本中提取处理意见信息
        disposal_opinion = []
        
        # 检查文本中是否包含"需要进行"或"进一步的"
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
            # 提取处理意见
            opinion = text[start:end].strip()
            if opinion:
                disposal_opinion.append(opinion)
        elif "进一步的" in text:
            start = text.find("进一步的") + 4
            # 查找下一个逗号、句号或"，治疗方案包括"
            end_markers = [",", "。", "，治疗方案包括"]
            end = len(text)
            for marker in end_markers:
                if marker in text[start:]:
                    temp_end = text.find(marker, start)
                    if temp_end < end:
                        end = temp_end
            # 提取处理意见
            opinion = text[start:end].strip()
            if opinion:
                disposal_opinion.append(opinion)
        
        # 检查文本中是否包含"治疗方案包括"
        if "治疗方案包括" in text:
            start = text.find("治疗方案包括") + 6
            # 查找句子结束
            if "。" in text[start:]:
                end = text.find("。", start)
                opinion = text[start:end].strip()
                if opinion:
                    disposal_opinion.append(opinion)
        
        # 检查文本中是否包含"开具了"
        if "开具了" in text:
            start = text.find("开具了") + 3
            # 查找句子结束
            if "。" in text[start:]:
                end = text.find("。", start)
                opinion = text[start:end].strip()
                if opinion:
                    disposal_opinion.append(opinion)
        
        # 去重
        disposal_opinion = list(set(disposal_opinion))
        
        # 如果没有从文本中提取到，使用实体信息
        if not disposal_opinion:
            # 添加治疗方法信息
            if 'TREATMENT' in entities:
                disposal_opinion.extend(entities['TREATMENT'])
            
            # 添加药物信息
            if 'MEDICATION' in entities:
                medications = entities['MEDICATION']
                if medications:
                    disposal_opinion.append(f"使用{', '.join(medications)}等药物治疗")
        
        # 如果仍然没有处理意见信息，返回"无"
        if not disposal_opinion:
            return "无"
        
        # 合并处理意见项目
        return '，'.join(disposal_opinion)
    
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
