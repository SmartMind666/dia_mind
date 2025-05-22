
import pandas as pd
import os

def process_sft1_train():
    queries = []
    answers = []
    for file in os.listdir('./qa'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel("./qa/"+file)
        input_column = list(df.columns)[-3]
        answer_column = list(df.columns)[-1]
        df = df.dropna(subset=[input_column, answer_column]).reset_index(drop=True)
        for idx in range(len(df)):
            queries.append(df.at[idx, input_column])
            answers.append(df.at[idx, answer_column])

    merged = pd.DataFrame({"query": queries, "answer": answers})
    merged.to_csv("./qa/qa_all.csv")

def process_sft1_test():
    queries = []
    answers = []
    for file in os.listdir('./qa_test'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel("./qa_test/" + file)
        input_column = list(df.columns)[-3]
        answer_column = list(df.columns)[-1]
        df = df.dropna(subset=[input_column, answer_column]).reset_index(drop=True)
        for idx in range(len(df)):
            if df.at[idx, input_column] is not None and df.at[idx, answer_column] is not None:
                queries.append(df.at[idx, input_column])
                answers.append(df.at[idx, answer_column])

    merged = pd.DataFrame({"query": queries, "answer": answers})
    merged.to_csv("./qa_test/qa_test.csv")

def process_sft2_train():
    queries = []
    answers = []
    for file in os.listdir('./diabetes_qa'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel("./diabetes_qa/" + file)
        input_column = list(df.columns)[-2]
        answer_column = list(df.columns)[-1]
        df = df.dropna(subset=[input_column, answer_column]).reset_index(drop=True)
        for idx in range(len(df)):
            if df.at[idx, input_column] is not None and df.at[idx, answer_column] is not None:
                queries.append(df.at[idx, input_column])
                answers.append(df.at[idx, answer_column])

    merged = pd.DataFrame({"query": queries, "answer": answers})
    merged.to_csv("./diabetes_qa/diabetes_qa.csv")

def process_sft2_test():
    queries = []
    answers = []
    for file in os.listdir('./diabetes_qa_test'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel("./diabetes_qa_test/" + file)
        input_column = list(df.columns)[-2]
        answer_column = list(df.columns)[-1]
        df = df.dropna(subset=[input_column, answer_column]).reset_index(drop=True)
        for idx in range(len(df)):
            if df.at[idx, input_column] is not None and df.at[idx, answer_column] is not None:
                queries.append(df.at[idx, input_column])
                answers.append(df.at[idx, answer_column])

    merged = pd.DataFrame({"query": queries, "answer": answers})
    merged.to_csv("./diabetes_qa_test/diabetes_qa_test.csv")


def process_reasoning_train():
    queries = []
    answers = []
    thinkings = []

    for file in os.listdir('./qa-reasoning'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel('./qa-reasoning/' + file)
        input_column = list(df.columns)[-3]
        thinking_column = list(df.columns)[-2]
        answer_column = list(df.columns)[-1]
        for idx in range(len(df)):
            queries.append(df.at[idx, input_column])
            answers.append(df.at[idx, answer_column])
            thinkings.append(df.at[idx, thinking_column])

    merged = pd.DataFrame({"query": queries, "thinking": thinkings, "answer": answers})
    merged.to_csv("./qa-reasoning/qa_reasoning_all.csv")

def process_reasoning_test():
    queries = []
    answers = []

    for file in os.listdir('./qa-reasoning-test'):
        if not file.endswith('.xlsx'):
            continue
        print(f'processing {file}')
        df = pd.read_excel('./qa-reasoning-test/' + file)
        input_column = list(df.columns)[-2]
        answer_column = list(df.columns)[-1]
        for idx in range(len(df)):
            queries.append(df.at[idx, input_column])
            answers.append(df.at[idx, answer_column])

    merged = pd.DataFrame({"query": queries, "answer": answers})
    merged.to_csv("./qa-reasoning-test/qa_reasoning_test.csv")
