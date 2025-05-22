import pandas as pd
import json

df = pd.read_excel("测试_medical打分表.xlsx", sheet_name="Sheet1")

for row in range(1500):
    prompt = f"""你是一位经验丰富的临床医学专家，具备深厚的专业知识和丰富的临床经验，能够从患者提供的症状和病史入手，进行系统性评估和缜密推理。以下是患者的问题以及四位不同医生的回答，请你在病理机制、鉴别诊断、辅助检查、治疗建议、并发症及危重情况提示、护理及日常生活建议、总体印象这7个方面对每位医生的回答进行评分。


# 患者的问题

{df.at[row, "问题"]}


# 医生A的回复

{df.at[row, "医生答案"]}


# 医生B的回复

{df.at[row, "Deepseekv3-withoutprompt"]}


# 医生C的回复

{df.at[row, "Deepseekv3-withprompt"]}


# 医生D的回复

{df.at[row, "DeepseekR1"]}


评分的规则为：每一方面的最高分都为10分，最低分都为0分。你需要严格按照以下的模板输出，每位医生的评分占用一行：

<医生编号>:<病理机制的分数>/<鉴别诊断的分数>/<辅助检查的分数>/<治疗建议的分数>/<并发症及危重情况提示的分数>/<护理及日常生活建议的分数>/<总体印象的分数>

例如：

A:8/7/9/8/8/10/9

必须包括对A、B、C、D四位医生的评价，**不要输出任何其他内容**"""
    request = {
        "custom_id": row,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "qwen-max-latest",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
    }

    with open("request.json", "a") as f:
        json_txt = json.dumps(request)
        f.write(json_txt + "\n")

