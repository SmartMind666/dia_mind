import json
import re

class_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

mark = [
    [0, 0, 0, 0, 0, 0, 0],      # 医生答案
    [0, 0, 0, 0, 0, 0, 0],      # Deepseekv3-withoutprompt
    [0, 0, 0, 0, 0, 0, 0],      # Deepseekv3-withprompt
    [0, 0, 0, 0, 0, 0, 0],      # DeepseekR1
]

json_raw_str = open("response.jsonl", "r").read()
json_list = json_raw_str.split("<|split_tag|>")
for row in json_list:
    response = json.loads(row)
    response_txt_list = response["response"]["body"]["choices"][0]["message"]["content"].split("\n")
    for mark_txt in response_txt_list:
        mark_splited_txt = re.match(r"([A-Z]):([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)/([0-9]*)", mark_txt)
        class_txt = mark_splited_txt.group(1)
        score_txt = [mark_splited_txt.group(i) for i in range(2, 9)]
        for i in range(7):
            mark[class_map[class_txt]][i] += int(score_txt[i])

for i in range(4):
    for j in range(7):
        mark[i][j] = float(mark[i][j]) / float(len(json_list))

print("========== 医生答案 ==========")
print(f"病理机制: {mark[0][0]}")
print(f"鉴别诊断: {mark[0][1]}")
print(f"辅助检查: {mark[0][2]}")
print(f"治疗建议: {mark[0][3]}")
print(f"并发症及危重情况提示: {mark[0][4]}")
print(f"护理及日常生活建议: {mark[0][5]}")
print(f"总体印象: {mark[0][6]}")
print(f"总分: {sum(mark[0]) / 7.0}")
print("")

print("========== Deepseekv3-withoutprompt ==========")
print(f"病理机制: {mark[1][0]}")
print(f"鉴别诊断: {mark[1][1]}")
print(f"辅助检查: {mark[1][2]}")
print(f"治疗建议: {mark[1][3]}")
print(f"并发症及危重情况提示: {mark[1][4]}")
print(f"护理及日常生活建议: {mark[1][5]}")
print(f"总体印象: {mark[1][6]}")
print(f"总分: {sum(mark[1]) / 7.0}")
print("")

print("========== Deepseekv3-withprompt ==========")
print(f"病理机制: {mark[2][0]}")
print(f"鉴别诊断: {mark[2][1]}")
print(f"辅助检查: {mark[2][2]}")
print(f"治疗建议: {mark[2][3]}")
print(f"并发症及危重情况提示: {mark[2][4]}")
print(f"护理及日常生活建议: {mark[2][5]}")
print(f"总体印象: {mark[2][6]}")
print(f"总分: {sum(mark[2]) / 7.0}")
print("")

print("========== DeepseekR1 ==========")
print(f"病理机制: {mark[3][0]}")
print(f"鉴别诊断: {mark[3][1]}")
print(f"辅助检查: {mark[3][2]}")
print(f"治疗建议: {mark[3][3]}")
print(f"并发症及危重情况提示: {mark[3][4]}")
print(f"护理及日常生活建议: {mark[3][5]}")
print(f"总体印象: {mark[3][6]}")
print(f"总分: {sum(mark[3]) / 7.0}")
print("")