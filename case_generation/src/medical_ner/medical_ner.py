import spacy

class MedicalNER:
    def __init__(self):
        # 加载医学NER模型
        try:
            # 尝试加载中文模型
            self.nlp = spacy.load('zh_core_web_sm')
        except OSError:
            try:
                # 尝试加载医学专业模型
                self.nlp = spacy.load('en_medical_ner')
            except OSError:
                # 如果医学模型不可用，使用英文通用模型
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                except OSError:
                    # 如果通用模型也不可用，下载并加载
                    spacy.cli.download('en_core_web_sm')
                    self.nlp = spacy.load('en_core_web_sm')
        
        # 定义医学实体类型
        self.medical_entity_types = {
            'DISEASE': '疾病',
            'SYMPTOM': '症状',
            'MEDICATION': '药物',
            'TREATMENT': '治疗方法',
            'BODY_PART': '身体部位',
            'TEST': '检查项目',
            'LAB_VALUE': '实验室值'
        }
    
    def recognize_entities(self, text):
        """
        识别文本中的医学实体
        :param text: 输入文本
        :return: 识别到的实体字典
        """
        # 处理文本
        doc = self.nlp(text)
        
        # 提取实体
        entities = {}
        for ent in doc.ents:
            # 检查实体类型是否为医学相关
            if ent.label_ in self.medical_entity_types:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
        
        # 如果使用的是通用模型，尝试基于关键词识别医学实体
        if len(entities) == 0:
            entities = self._keyword_based_recognition(text)
        
        return entities
    
    def _keyword_based_recognition(self, text):
        """
        基于关键词的医学实体识别
        :param text: 输入文本
        :return: 识别到的实体字典
        """
        # 简单的关键词列表，实际应用中可以扩展
        keywords = {
            'DISEASE': ['diabetes', 'hypertension', 'cancer', 'asthma', 'heart disease', 'stroke', '高血压', '糖尿病', '癌症', '哮喘', '心脏病', '中风', '溃疡性结肠炎'],
            'SYMPTOM': ['pain', 'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', '头痛', '发热', '咳嗽', '疲劳', '恶心', '呕吐', '腹痛', '腹泻'],
            'MEDICATION': ['aspirin', 'ibuprofen', 'acetaminophen', 'antibiotic', 'insulin', '抗生素', '糖皮质激素'],
            'TREATMENT': ['surgery', 'therapy', 'medication', 'exercise', 'diet', '手术', '治疗', '药物治疗'],
            'BODY_PART': ['heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', '心脏', '肺', '肝', '肾', '脑', '胃'],
            'TEST': ['blood test', 'x-ray', 'ct scan', 'mri', 'ultrasound', '血液检查', '结肠镜检查', '粪便常规检查'],
            'LAB_VALUE': ['blood pressure', 'blood sugar', 'cholesterol', 'temperature', '血压', '血糖', '胆固醇', '体温']
        }
        
        entities = {}
        
        for entity_type, entity_keywords in keywords.items():
            for keyword in entity_keywords:
                if keyword in text:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(keyword)
        
        return entities
    
    def get_entity_summary(self, entities):
        """
        获取实体摘要
        :param entities: 识别到的实体字典
        :return: 实体摘要字符串
        """
        summary = ""
        for entity_type, entity_list in entities.items():
            if entity_list:
                chinese_type = self.medical_entity_types.get(entity_type, entity_type)
                summary += f"{chinese_type}: {', '.join(entity_list)}\n"
        return summary
