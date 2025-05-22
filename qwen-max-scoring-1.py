import pandas as pd
from openai import OpenAI

df = pd.read_excel("dataset-7.xlsx", sheet_name="运动饮食与日常管理")
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

if not "qwen-max output" in df.columns:
    df.insert(df.shape[1], "qwen-max output", None)
    df = df.astype({"qwen-max output": "object"})

client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

for row in range(df.shape[0]):
    prompt = f"""你是一位经验丰富的临床医学专家，具备深厚的专业知识和丰富的临床经验，能够从患者提供的症状和病史入手，进行系统性评估和缜密推理。以下是患者的问题以及四位不同医生的回答，请你在医学准确性、信息全面性、个体化适配度、时效性验证、风险预警完整性、患者可及性这6个方面对每位医生的回答进行评分。


# 患者的问题

{df.at[row, "question"]}


# 医生A的回复

{df.at[row, "v3-without-prompt"]}


# 医生B的回复

{df.at[row, "v3-with-prompt"]}


# 医生C的回复

{df.at[row, "r1-with-prompt"]}


# 医生D的回复

{df.at[row, "r1-with-rag"]}


# 评价方法与指标

1. 医学准确性
评价指标：是否符合糖尿病最新标准
2. 信息全面性
评价指标：是否覆盖"评估-诊断-治疗-监测-预后"全链条
3. 个体化适配度
评价指标：是否体现精准医疗分层原则
4. 时效性验证
评估方法：是否包含前沿研究和软硬件设备
5. 风险预警完整性
评估方法：是否包含药物不良反应警示，运动禁忌证识别，低血糖高血糖危险处理
6. 患者可及性
评估方法：治疗方案成本效益分析，地域医疗资源适配

评分的规则为：每一方面的最高分都为10分，最低分都为0分。你需要严格按照以下的模板输出，每位医生的评分占用一行：

<医生编号>:<医学准确性的分数>/<信息全面性的分数>/<个体化适配度的分数>/<时效性验证的分数>/<风险预警完整性的分数>/<患者可及性的分数>

例如：

A:8/7/9/8/8/10

必须包括对A、B、C、D四位医生的评价，**不要输出任何其他内容**"""

    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    df.at[row, "qwen-max output"] = completion.choices[0].message.content
    df.to_excel("dataset-7-output-运动饮食.xlsx", index=False)
    print(f"{row + 1} / {df.shape[0]}")

