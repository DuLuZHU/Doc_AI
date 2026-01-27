import click
from src.text_processing.text_processor import TextProcessor
from src.medical_ner.medical_ner import MedicalNER
from src.case_generation.case_generator import CaseGenerator

@click.command()
def cli():
    """
    医学病例自动生成工具
    """
    click.echo("=== 医学病例自动生成工具 ===")
    click.echo("请输入一段描述症状、疾病或治疗的文本，系统将自动生成结构化病例。")
    click.echo("例如：患者因头痛、发热3天入院，既往有高血压病史，需要进行血液检查。")
    click.echo("输入 'exit' 退出程序。")
    click.echo()
    
    # 初始化各模块
    text_processor = TextProcessor()
    medical_ner = MedicalNER()
    case_generator = CaseGenerator()
    
    while True:
        # 获取用户输入
        input_text = click.prompt("请输入文本", type=str)
        
        if input_text.lower() == 'exit':
            click.echo("程序已退出。")
            break
        
        if not input_text.strip():
            click.echo("输入不能为空，请重新输入。")
            continue
        
        # 处理输入文本
        processed_text, key_phrases = text_processor.process_input(input_text)
        
        # 识别医学实体
        entities = medical_ner.recognize_entities(processed_text)
        
        # 生成结构化病例
        case = case_generator.generate_case(input_text, entities)
        
        # 格式化并输出病例
        formatted_case = case_generator.format_case(case)
        
        click.echo("\n=== 生成的结构化病例 ===")
        click.echo(formatted_case)
        click.echo("====================\n")

if __name__ == "__main__":
    cli()
