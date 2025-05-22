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


def get_sft1_train_dataset(tokenizer):
    dataset = load_dataset("csv", data_files='./qa/qa_all.csv')

    # Function to convert Dolly format to conversation format
    def qa_to_conversation(example):
        conversation = f"<|im_start|>user\n{example['query']}<|im_end|>\n" + f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
        return {"conversations": conversation}

    # Convert the dataset
    dataset = dataset.map(qa_to_conversation)

    # Function to format conversations using the tokenizer
    def formatting_prompts_func(examples):

        convos = examples["conversations"]
        texts = []
        for convo in convos:
            try:
                texts.append(convo)
            except Exception as e:
                print(f'error while processing: {convo}, error: {e}')
        return {"text": texts}

    # Apply the formatting function to the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset['train'], tokenizer


def get_sft2_train_dataset(tokenizer):
    dataset = load_dataset("csv", data_files='./diabetes_qa/diabetes_qa.csv')

    # Function to convert Dolly format to conversation format
    def qa_to_conversation(example):
        conversation = f"<|im_start|>user\n{example['query']}<|im_end|>\n" + f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
        return {"conversations": conversation}

    # Convert the dataset
    dataset = dataset.map(qa_to_conversation)

    # Function to format conversations using the tokenizer
    def formatting_prompts_func(examples):

        convos = examples["conversations"]
        texts = []
        for convo in convos:
            try:
                texts.append(convo)
            except Exception as e:
                print(f'error while processing: {convo}, error: {e}')
        return {"text": texts}

    # Apply the formatting function to the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset['train'], tokenizer



from bert_score import score as bert_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(model, tokenizer, dataset_path, generate_response_func, max_new_tokens=2048, model_name=""):
    FastLanguageModel.for_inference(model)
    # We load the entire dataset as 'train' split:
    test_dataset = load_dataset("csv", data_files=dataset_path)['train']
    dataset = test_dataset.train_test_split(test_size=100, seed=3834)
    test_dataset = dataset['test']
    results = {
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
    def generate_response(query, context):
      messages = [{"role":"system", "content":"你是一位经验丰富的临床医学专家，具备深厚的专业知识和丰富的临床经验，能够基于患者描述的症状和病史，提供科学、合理的医疗建议"},{"role": "user", "content": query}]
      inputs = tokenizer.apply_chat_template(
          messages,
          tokenize=True,
          add_generation_prompt=True,
          return_tensors="pt"
      ).to("cuda")

      outputs = model.generate(
          input_ids=inputs,
          max_new_tokens=max_new_tokens,
          use_cache=True,
          temperature=0.9,
          num_return_sequences=1
      )
      res = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
      return res

    def generate_response_without_instrct(query, context):
        messages = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(
            [messages],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs["input_ids"].to("cuda"),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.9,
            num_return_sequences=1
        )
        res = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return res


    # Evaluate the model
    gr_func = {
        "generate_response": generate_response,
        "generate_response_without_instrct": generate_response_without_instrct
    }
    for example in tqdm(test_dataset, 'evaluating..'):
        query = example['query']
        answer = example['answer']
        context = example.get('context', "")

        # Generate model's response
        generated = gr_func[generate_response_func](query, context)
        # print(generated)
        # print(answer)

        # Calculate ROUGE scores
        scores = scorer.score(answer, generated)
        results["rouge1"].append(scores['rouge1'].fmeasure)
        results["rouge2"].append(scores['rouge2'].fmeasure)
        results["rougeL"].append(scores['rougeL'].fmeasure)
        results["predictions"].append(generated)
        results["references"].append(answer)

    metrics = {
        "rouge1": np.nanmean(results["rouge1"]),
        "rouge2": np.nanmean(results["rouge2"]),
        "rougeL": np.nanmean(results["rougeL"]),
    }

    # BLEU计算（带平滑处理）
    valid_pairs = [(p, r) for p, r in zip(results["predictions"], results["references"]) if p and r]
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
        "reference": results["references"],
        "prediction": results["predictions"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
    })
    try:
        detail_df.to_excel(f"{model_name}_evaluation_details.xlsx", index=False, engine='openpyxl')
    except Exception as e:
        print(f"\n结果保存失败：{str(e)}")

    return metrics


wandb.ensure_configured()
from unsloth import FastLanguageModel


def train_sft1(model_path, model_name):
    import torch
    max_seq_length = 8092
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3834,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    dataset, tokenizer = get_sft1_train_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            # max_steps = 900,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",  # Use this for WandB etc
            run_name=f"{model_name}_sft1"
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[2]["labels"]]))
    wandb.login(key="b53ad07343c259b3da79009271ce7e7b854dd637")
    with wandb.init(project='diabetes', name=f"{model_name}_sft1") as run:
        trainer_stats = trainer.train()
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    model.save_pretrained_merged(
        save_directory=f"{model_path}/{model_name}_sft1",  # 指定保存路径
        tokenizer=tokenizer,
        save_method="merged_16bit")
    evaluate(model, tokenizer, generate_response_func='generate_response_without_instrct', dataset_path='./qa_test/qa_test.csv',
             model_name=model_name)


def train_sft2(model_path, model_name):
    import torch
    max_seq_length = 8092
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{model_path}/{model_name}_sft1",
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3834,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    dataset, tokenizer = get_sft2_train_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            # max_steps = 900,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",  # Use this for WandB etc
            run_name=f"{model_name}_sft2"
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[2]["labels"]]))
    wandb.login(key="b53ad07343c259b3da79009271ce7e7b854dd637")
    with wandb.init(project='diabetes', name=f"{model_name}_sft2") as run:
        trainer_stats = trainer.train()
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    model.save_pretrained_merged(
        save_directory=f"{model_path}/ckpts/{model_name}_sft2",  # 指定保存路径
        tokenizer=tokenizer,
        save_method="merged_16bit")
    evaluate(model, tokenizer, generate_response_func='generate_response_without_instrct',
             dataset_path='./diabetes_qa_test/diabetes_qa_test.csv', model_name=f"{model_name}_sft2")

def train_sft3(model_path, model_name):
    import torch
    max_seq_length = 8092
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3834,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    dataset, tokenizer = get_sft2_train_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,  # Set this for 1 full training run.
            # max_steps = 900,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",  # Use this for WandB etc
            run_name=f"{model_name}_sft2_without_sft1"
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[2]["labels"]]))
    wandb.login(key="b53ad07343c259b3da79009271ce7e7b854dd637")
    with wandb.init(project='diabetes', name=f"{model_name}_sft2_without_sft1") as run:
        trainer_stats = trainer.train()
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    model.save_pretrained_merged(
        save_directory=f"{model_path}/{model_name}_sft2_without_sft1",  # 指定保存路径
        tokenizer=tokenizer,
        save_method="merged_16bit")
    res = evaluate(model, tokenizer, generate_response_func='generate_response_without_instrct',
             dataset_path='./diabetes_qa_test/diabetes_qa_test.csv', model_name=f"{model_name}_sft2_without_sft1")
    print(res)

# code to train qwen2.5 series with no instruct for sft1, sft2, sft3
train_sft1(model_path="/root/autodl-tmp/unsloth", model_name='Qwen2.5-7B')
train_sft2(model_path="/root/autodl-tmp/unsloth", model_name='Qwen2.5-7B')
train_sft3(model_path="/root/autodl-tmp/unsloth", model_name='Qwen2.5-7B')