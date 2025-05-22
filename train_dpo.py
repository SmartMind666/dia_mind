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

def get_dpo_train_dataset(tokenizer):
    dataset = load_dataset("csv", data_files='./dpo/dpo.csv')
    rejected_list = ["Deepseekv3-withoutprompt"]
    def qa_to_conversation(example):
        rejected = np.random.choice(rejected_list, 1)[0]
        # 构造完整对话结构
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": ""},  # 空内容标记回答开始位置
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenization处理
        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        
        tokenized_chosen = tokenizer(
            example["DeepseekR1"] + "<|im_end|>",  # 确保结束标记
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        
        tokenized_rejected = tokenizer(
            example[rejected] + "<|im_end|>",
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        
        return {
            # 原始文本字段
            "prompt": prompt,
            "chosen": example["DeepseekR1"] + "<|im_end|>",
            "rejected": example[rejected] + "<|im_end|>",
            
            # Tokenized字段（必须为int64类型）
            "prompt_input_ids": tokenized_prompt["input_ids"][0].numpy().tolist(),
            "prompt_attention_mask": tokenized_prompt["attention_mask"][0].numpy().tolist(),
            "chosen_input_ids": tokenized_chosen["input_ids"][0].numpy().tolist(),
            "chosen_attention_mask": tokenized_chosen["attention_mask"][0].numpy().tolist(),
            "rejected_input_ids": tokenized_rejected["input_ids"][0].numpy().tolist(),
            "rejected_attention_mask": tokenized_rejected["attention_mask"][0].numpy().tolist(),
        }

    # Convert the dataset
    dataset = dataset.map(qa_to_conversation)

    # Function to format conversations using the tokenizer
    # def formatting_prompts_func(examples):
    #     convos = examples["conversations"]
    #     texts = [tokenizer.apply_chat_template(convo, tokenize=False) for convo in convos]
    #     return {"text": texts}

    # Apply the formatting function to the dataset
    # dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset['train'], tokenizer

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

wandb.ensure_configured()
from unsloth import FastLanguageModel

def train_dpo(model_path, model_name):
    max_seq_length = 8092
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{model_path}/{model_name}_think_reasoning",
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_alpha=16,
        lora_dropout=0.05,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3834,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    dataset, tokenizer = get_dpo_train_dataset(tokenizer)

    from trl import DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=3e-6,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            output_dir="dpo_output",
            optim="adafactor",
            report_to="wandb",  # Use this for WandB etc
            run_name=f"{model_name}_dpo"
        ),
        beta=0.2,  # DPO温度参数
        train_dataset=dataset,
        # eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        # generate_during_eval=True,
        loss_type="sigmoid",  # 明确损失类型
    )

    with wandb.init(project='diabetes', name=f"{model_name}_dpo") as run:
        trainer_stats = trainer.train()
    model.save_pretrained_merged(
        save_directory=f"{model_path}/{model_name}_dpo",  # 指定保存路径
        tokenizer=tokenizer,
        save_method="merged_16bit")
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    res = evaluate(model, tokenizer, model_name=f"{model_name}_dpo")
    print(res)

train_dpo("/root/autodl-tmp/unsloth", "Qwen-7B-Instruct")