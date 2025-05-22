import pandas as pd
import re

class_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

mark = [
    [0, 0, 0, 0, 0, 0],      # v3-without-prompt
    [0, 0, 0, 0, 0, 0],      # v3-with-prompt
    [0, 0, 0, 0, 0, 0],      # r1-with-prompt
    [0, 0, 0, 0, 0, 0],      # r1-with-rag
]

df = pd.read_excel("dataset-7-output-运动饮食.xlsx", sheet_name="Sheet1")
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

for row in range(df.shape[0]):
    response_txt_list = df.at[row, "qwen-max output"].split("\n")
    for mark_txt in response_txt_list:
        mark_splited_txt = re.match(r"([A-Z]):([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)", mark_txt)
        class_txt = mark_splited_txt.group(1)
        score_txt = [mark_splited_txt.group(i) for i in range(2, 8)]
        for i in range(6):
            mark[class_map[class_txt]][i] += int(score_txt[i])

for i in range(4):
    for j in range(6):
        mark[i][j] = float(mark[i][j]) / 10.0

print("========== v3-without-prompt ==========")
print(f"医学准确性: {mark[0][0]}")
print(f"信息全面性: {mark[0][1]}")
print(f"个体化适配度: {mark[0][2]}")
print(f"时效性验证: {mark[0][3]}")
print(f"风险预警完整性: {mark[0][4]}")
print(f"患者可及性: {mark[0][5]}")
print(f"总分: {sum(mark[0]) / 6.0}")
print("")

print("========== v3-with-prompt ==========")
print(f"医学准确性: {mark[1][0]}")
print(f"信息全面性: {mark[1][1]}")
print(f"个体化适配度: {mark[1][2]}")
print(f"时效性验证: {mark[1][3]}")
print(f"风险预警完整性: {mark[1][4]}")
print(f"患者可及性: {mark[1][5]}")
print(f"总分: {sum(mark[1]) / 6.0}")
print("")

print("========== r1-with-prompt ==========")
print(f"医学准确性: {mark[2][0]}")
print(f"信息全面性: {mark[2][1]}")
print(f"个体化适配度: {mark[2][2]}")
print(f"时效性验证: {mark[2][3]}")
print(f"风险预警完整性: {mark[2][4]}")
print(f"患者可及性: {mark[2][5]}")
print(f"总分: {sum(mark[2]) / 6.0}")
print("")

print("========== r1-with-rag ==========")
print(f"医学准确性: {mark[3][0]}")
print(f"信息全面性: {mark[3][1]}")
print(f"个体化适配度: {mark[3][2]}")
print(f"时效性验证: {mark[3][3]}")
print(f"风险预警完整性: {mark[3][4]}")
print(f"患者可及性: {mark[3][5]}")
print(f"总分: {sum(mark[3]) / 6.0}")
print("")