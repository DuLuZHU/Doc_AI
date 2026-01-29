from src.case_generation.case_generator import CaseGenerator
from src.medical_ner.medical_ner import MedicalNER

# 初始化CaseGenerator和MedicalNER
case_generator = CaseGenerator()
medical_ner = MedicalNER()

# 复杂测试用例
test_text5 = "患者姓名张三，女性，45岁，因腹痛、腹泻3天入院，伴有发热，体温38.5℃，进行血检检查，报告显示白细胞数为10*10^20，白细胞异常有发炎症状。既往有溃疡性结肠炎病史，需要进一步的结肠镜检查和粪便常规检查，治疗方案包括使用抗生素和糖皮质激素，开具了头孢，每次100mg，每日2次。"

print("=== 测试复杂病例描述 ===")
print("输入文本:")
print(test_text5)
print("\n输出结果:")
entities5 = medical_ner.recognize_entities(test_text5)
case5 = case_generator.generate_case(test_text5, entities5)
print(case_generator.format_case(case5))
