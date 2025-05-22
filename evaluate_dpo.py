import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                        text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
import os

os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import wandb
from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import pandas as pd
import numpy as np

def get_dpo_test_dataset(tokenizer):
    dataset = load_dataset("csv", data_files='./dpo_test/dpo_test.csv')
    def qa_to_conversation(example):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": ""},  # 空内容标记回答开始位置
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenization处理
        
        
        
        return {
            # 原始文本字段
            "prompt": prompt,
            "chosen": example["DeepseekR1"] + "<|im_end|>",
            "rejected": example['Deepseekv3-withoutprompt'] + "<|im_end|>",
        }

    # Convert the dataset
    dataset = dataset.map(qa_to_conversation)

    return dataset['train'], tokenizer

def get_dpo_test_dataset1(tokenizer):
    dataset = load_dataset("csv", data_files='./dpo_test/dpo_test.csv')
    # def qa_to_conversation(example):
    #     rejected = 'Deepseekv3-withoutprompt'
    #     formatted_prompt = f"<|im_start|>user\n{example['query']}<|im_end|>\n<|im_start|>assistant\n"
    #     return {"query": formatted_prompt, "chosen": example["DeepseekR1"] + "<|im_end|>", "rejected": example[rejected] + "<|im_end|>"}

    # # Convert the dataset
    # dataset = dataset.map(qa_to_conversation)

    return dataset['train'], tokenizer


from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(model, tokenizer, max_new_tokens=2048, model_name=""):
    FastLanguageModel.for_inference(model)
    # We load the entire dataset as 'train' split:
    test_dataset = get_dpo_test_dataset(tokenizer)
    results_with_rejected = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "predictions": [],
        "references": []
    }

    results_with_chosen = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "predictions": [],
        "references": []
    }

    # We then create a test split from this data.
    # This approach creates a test set that's 5% of the original data, while 95% remains for training.
    # The seed=4016 ensures reproducibility of the split.
    # dataset = dataset.train_test_split(test_size=1200, seed=3834)
    # test_dataset = dataset['test']

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Function to generate response

    def cal_score(result_container, answer, generated):
        scores = scorer.score(answer, generated)
        result_container["rouge1"].append(scores['rouge1'].fmeasure)
        result_container["rouge2"].append(scores['rouge2'].fmeasure)
        result_container["rougeL"].append(scores['rougeL'].fmeasure)
        result_container["predictions"].append(generated)
        result_container["references"].append(answer)
    queries = test_dataset[0]['prompt']
    chosens = test_dataset[0]['chosen']
    rejecteds = test_dataset[0]['rejected']
    for idx in tqdm(range(len(queries)), 'evaluating...'):
        query = queries[idx]
        chosen = chosens[idx]
        rejected = rejecteds[idx]
        tokenized_prompt = tokenizer(
            query,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        input_ids = tokenized_prompt["input_ids"].to("cuda")
        attention_mask = tokenized_prompt["attention_mask"].to("cuda")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.9,
            num_return_sequences=1
        )
        generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print(f"generate: {generated}")
        # print(f"query: {query[:100]}, len: {len(query)}")
        # print(f"chosen: {chosen[0][:100]}, len: {len(chosen)}")
        # print(f"rejected: {rejected[0][:100]}, len: {len(rejected)}")

        cal_score(results_with_chosen, chosen, generated)
        cal_score(results_with_rejected, rejected, generated)
    def cal_save_metrics(result_container, name):

        metrics = {
            "rouge1": np.nanmean(result_container["rouge1"]),
            "rouge2": np.nanmean(result_container["rouge2"]),
            "rougeL": np.nanmean(result_container["rougeL"]),
        }

        # BLEU计算（带平滑处理）
        valid_pairs = [(p, r) for p, r in zip(result_container["predictions"], result_container["references"]) if p and r]
        if valid_pairs:
            smoothie = SmoothingFunction().method1
            metrics["bleu"] = corpus_bleu(
                [[r.split()] for _, r in valid_pairs],  # 二维列表结构
                [p.split() for p, _ in valid_pairs],
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothie
            )
        else:
            metrics["bleu"] = 0.0

            # BERTScore计算（修复中文检测）
        if valid_pairs:
            has_chinese = False
            for p, _ in valid_pairs:
                if any('\u4e00' <= char <= '\u9fff' for char in p):
                    has_chinese = True
                    break
            lang = "zh" if has_chinese else "en"

            P, R, F1 = bert_score(
                [p for p, _ in valid_pairs],
                [r for _, r in valid_pairs],
                lang=lang,
                rescale_with_baseline=True
            )
            metrics.update({
                "bert_score_precision": P.mean().item(),
                "bert_score_recall": R.mean().item(),
                "bert_score_f1": F1.mean().item(),
            })
        else:
            metrics.update({
                "bert_score_precision": 0.0,
                "bert_score_recall": 0.0,
                "bert_score_f1": 0.0,
            })

        # 保存结果
        detail_df = pd.DataFrame({
            "reference": result_container["references"],
            "prediction": result_container["predictions"],
            "rouge1": result_container["rouge1"],
            "rouge2": result_container["rouge2"],
            "rougeL": result_container["rougeL"],
        })
        try:
            detail_df.to_excel(f"{model_name}_{name}_evaluation_details.xlsx", index=False, engine='openpyxl')
        except Exception as e:
            print(f"\n结果保存失败：{str(e)}")
        return metrics
    mc = cal_save_metrics(results_with_chosen, "chosen")
    mr = cal_save_metrics(results_with_rejected, "rejected")
    return mc, mr

from unsloth import FastLanguageModel

def evaluate_dpo(model_path="unsloth/Qwen2.5-7B-Instruct"):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=8092,
    dtype=None,
    local_files_only=True
    )
    FastLanguageModel.for_inference(model)
    res = evaluate(model, tokenizer, max_new_tokens=4096, model_name=model_path.split("/")[-1])
    print(res)
    
evaluate_dpo()
evaluate_dpo("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft1")
evaluate_dpo("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft2")
evaluate_dpo("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft2_without_sft1")
evaluate_dpo("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_think_reasoning")
evaluate_dpo("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_dpo")