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

import re
def extract_content(text):
  pattern = r'</think>\s*([^\s].*?)\s*<\|im_end\|>'
  match = re.search(pattern, text)
  if match:
      content = match.group(1).strip()
      return content
  else:
      # 如果没有找到匹配的<|im_end|>，提取</think>后面的所有非空白字符开始的内容
      pattern_all = r'</think>\s*([^\s].*)'
      match_all = re.search(pattern_all, text)
      if match_all:
          return match_all.group(1).strip()
      else:
          return text


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

    def generate_response_reasoning(query, context):
        messages = [{"role": "system",
                     "content": "你是一位经验丰富的临床医学专家，具备深厚的专业知识和丰富的临床经验，能够从患者提供的症状和病史入手，进行系统性评估和缜密推理。通过对病情的逐步剖析，，最终为患者提供科学、合理且个性化的医疗建议"},
                    {"role": "user", "content": query}]
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
        res = extract_content(res)
        return res

    def generate_response_reasoning_rag(query, context):
        messages = [{"role": "system",
                     "content": f"""你是一位经验丰富的临床医学专家，具备深厚的专业知识和丰富的临床经验，能够从患者提供的症状和病史入手，结合提供的资料和自身知识，进行系统性评估和缜密推理。通过对病情的逐步剖析，，最终为患者提供科学、合理且个性化的医疗建议。以下是可供参考的资料：\n{context}"""},
                    {"role": "user", "content": query}]
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
        res = extract_content(res)
        return res

    # Evaluate the model
    gr_func = {
        "generate_response": generate_response,
        "generate_response_reasoning": generate_response_reasoning,
        "generate_response_reasoning_rag": generate_response_reasoning_rag,
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


from unsloth import FastLanguageModel

def evaluate_sft123(model_path="Qwen/Qwen2.5-7B-Instruct", dataset_path='./qa_test/qa_test.csv', generate_response_func='generate_response'):

    max_seq_length = 8092
    dtype = None
    model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = model_path,
      max_seq_length = max_seq_length,
      dtype = dtype
    )

    res = evaluate(model, tokenizer, generate_response_func=generate_response_func, dataset_path=dataset_path, model_name=model_path.split("/")[-1])

    print(res)

# evaluate instruct
evaluate_sft123()
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft1")
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft2")
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B-Instruct_sft2_without_sft1")


# evaluate no instruct
evaluate_sft123("unsloth/Qwen2.5-7B", generate_response_func='generate_response_without_instrct')
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B_sft1", generate_response_func='generate_response_without_instrct')
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B_sft2", generate_response_func='generate_response_without_instrct')
evaluate_sft123("/root/autodl-tmp/unsloth/Qwen2.5-7B_sft2_without_sft1", generate_response_func='generate_response_without_instrct')