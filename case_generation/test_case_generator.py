from src.case_generation.case_generator import CaseGenerator
from src.medical_ner.medical_ner import MedicalNER

# 初始化CaseGenerator和MedicalNER
case_generator = CaseGenerator()
medical_ner = MedicalNER()

# 测试示例1
print("=== 测试示例1 ===")
test_text1 = "患者因头痛、发热3天入院，伴有恶心、呕吐症状，既往有高血压病史，需要进行血液检查。"
entities1 = medical_ner.recognize_entities(test_text1)
case1 = case_generator.generate_case(test_text1, entities1)
print(case_generator.format_case(case1))

# 测试示例2
print("\n=== 测试示例2 ===")
test_text2 = "患者因腹痛、腹泻3天入院，伴有发热，体温38.5℃，既往有溃疡性结肠炎病史，需要进行结肠镜检查和粪便常规检查，治疗方案包括使用抗生素和糖皮质激素。"
entities2 = medical_ner.recognize_entities(test_text2)
case2 = case_generator.generate_case(test_text2, entities2)
print(case_generator.format_case(case2))
