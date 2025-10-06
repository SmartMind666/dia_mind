# DiaMind: A Comprehensive Medical Dataset and Benchmark for Diabetes Management

**DiaMind** is the first integrated diabetes management framework combining:
-  🧠 **DiaData**: Comprehensive dataset for diabetes understanding
-  📊 **Diabench**: Multi-dimensional evaluation benchmark
-   **Multi-stage training framework**: Enhanced reasoning for clinical care

##  🔍 Overview
Diabetes affects **10.2%** of global adults (IDF 2030 projection) with inadequate management increasing complication risks. DiaMind bridges critical gaps in diabetes care AI by providing:
- ✅ Clinically validated QA pairs
-  🧬 Interdisciplinary knowledge integration
-  📈 Enhanced reasoning capabilities
-   Real-world clinical applicability

<img width="2228" height="992" alt="image" src="https://github.com/user-attachments/assets/8be33cc1-0397-4d06-9f40-fb098da42512" />

## ✨ Key Contributions
### 1. DiaData: Integrated Diabetes Dataset
| Component | Description | Size |
|-----------|-------------|------|
| **DiaRAG** | Retrieval-augmented resources | 289 guidelines/articles |
| **DiaQA** | Clinically validated QA pairs | 541,606 pairs |
| **Answer Types** | - Direct responses<br>- CoT reasoning traces<br>- Reward-optimized outputs | 542k / 53k / 7.8k |

### 2. Diabench: Evaluation Benchmark
- **7 quantitative metrics** for daily management
- **Multi-stage assessment**:
1. Supervised Fine-Tuning (SFT)
2. Reasoning Alignment
3. RLHF Optimization
- **Clinical evaluation dimensions**:
- Pathology • Diagnostics • Treatment • Complications • Patient Care

### 3. Multi-stage Training Framework
```mermaid
flowchart LR
S1[Base LLM] --> S2[SFT Training]
S2 --> S3[Reasoning Alignment]
S3 --> S4[DPO Optimization]
S4 --> S5[Diabetes Specialist Model]
```

## 📊 Key Results
### Data Quality Improvement
<img width="1440" height="554" alt="image" src="https://github.com/user-attachments/assets/29f5e96c-0e4d-49e5-a8ed-f0291ef38a58" />
Experimental results demonstrate that DiaMind-generated answers significantly outperform online doctor responses and zero-shot LLM outputs across all clinical evaluation dimensions.

### Model Performance
Through systematic experimentation, this study validates the efficacy of the multi-stage training framework and DiaMind dataset in
enhancing LLMs’ diabetes management capabilities, revealing intricate interactions among domain adaptation, model architecture,
and training strategies. Detailed experimental results are shown below.
#### general medical subset

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/2e98008c-f6bd-48e0-abcf-754aafcc425a" />

#### diabetes medical subset

<div align="center">
    <img width="700" alt="image" src="https://github.com/user-attachments/assets/f81d7265-50e0-4bcf-90ff-ef9cb3503737" />
</div>

#### Reasoning subset (Qwen-7b-instruction)

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/f2af83d6-4059-4061-a1d8-dbf48af77b15" />
</div>

#### DPO subset (Qwen-7b-instruction)

<div align="center">
    <img width="700" alt="image" src="https://github.com/user-attachments/assets/7a4283a5-14ed-4477-8b81-69db4e4e91bf" />
</div>

## **Dataset Availability**
The dataset is available for download [here](#) ([huggingface](https://huggingface.co/datasets/SmartMind666/dia_mind)).
