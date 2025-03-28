# InternTA: An Multi-agent AI Teaching Assistant Learns from Limited Data

[中文版](README.md) | [English Version](README-EN.md)

<div align="center"><img src="./statics/demo.gif" width="500"></div>

## Abstract

Large language models (LLMs) have shown great potential to enhance student learning by serving as AI-powered teaching assistants (TA). However, existing LLM-based TA systems often face critical challenges, including data privacy risks associated with third-party API-based solutions and limited effectiveness in courses with limited teaching materials.

This project proposes an automated TA training system based on LLM agents, designed to train customized, lightweight, and privacy-preserving AI models. Unlike traditional cloud-based AI TAs, our system allows local deployment, reducing data security concerns, and includes three components:

1. **Dataset Agent**: Constructing high-quality datasets with explicit reasoning paths
2. **Training Agent**: Fine-tuning models via Knowledge Distillation, effectively adapting to limited-data courses
3. **RAG Agent**: Enhancing responses by retrieving external knowledge

We validate our system in Synthetic Biology, an interdisciplinary field characterized by scarce structured training data. Experimental results and user evaluations demonstrate that our AI TA achieves strong performance, high user satisfaction, and improved student engagement, highlighting its practical applicability in real-world educational settings.

## Background

Synthetic biology is a cutting-edge field that integrates knowledge from biology, chemistry, engineering, and computer science. In recent years, applications ranging from lab-grown meat to CRISPR-Cas9 gene editing technology have been leading the "Third Biotechnology Revolution." However, the dissemination of synthetic biology knowledge faces two major challenges:

1. Interdisciplinary complexity: Requires integration of knowledge from multiple domains, creating a steep learning curve
2. Educational resource limitations: Shortage of teaching talent with cross-disciplinary knowledge and practical experience

Traditional AI teaching assistant solutions typically rely on cloud service APIs, which introduce data privacy risks and perform poorly when specialized teaching materials are limited. The InternTA project is designed to address these challenges.

## Technical Architecture

InternTA adopts a three-layer agent architecture to achieve automated training, local deployment, and privacy protection:

<div align="center"><img src="./statics/internTA.png" width="500"></div>

### 1. Dataset Agent

The Dataset Agent is responsible for constructing high-quality training data with explicit reasoning paths:

<div align="center"><img src="./statics/data-EN.png" width="350"></div>

- **Data Sources**: Extracts post-class questions, key terms, and fundamental concepts from the "Synthetic Biology" textbook
- **Reasoning Path Construction**: Generates explicit reasoning paths for each question
- **Guided Teaching Design**: For complex thought questions, designs guided responses rather than providing direct answers

### 2. Training Agent

The Training Agent fine-tunes lightweight models using knowledge distillation techniques:

- **Base Model**: Uses [DeepSeekR1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) as the foundation model
- **Fine-Tuning Tools**: Employs [PeftModel](https://github.com/huggingface/peft) for efficient fine-tuning
- **Knowledge Distillation**: Transfers knowledge from larger parameter-scale models to lightweight models

### 3. RAG Agent

The RAG (Retrieval-Augmented Generation) Agent enhances answer quality by retrieving external knowledge:

- **Knowledge Base Construction**: Structured processing of "Synthetic Biology" textbook content
- **Semantic Retrieval**: Retrieves relevant knowledge points based on user questions
- **Enhanced Generation**: Combines retrieved knowledge to generate more accurate and in-depth answers

## Privacy Protection and Local Deployment

InternTA system design emphasizes data privacy protection and deployment flexibility:

- **Local Model Deployment**: All models can run on local machines, avoiding data exposure
- **API Token Authentication**: Provides API access control mechanisms to secure the system
- **Lightweight Design**: Optimizes model size to run efficiently on ordinary hardware

## Quick Experience

**Online Experience Address**: [[Powered by Coze]](https://www.kongfoo.cloud/)

**Local Deployment Method** (NVIDIA GPU with 8GB or more VRAM):

```sh
# Clone the repository
git clone https://github.com/kongfoo-ai/internTA

# Go to the project directory
cd InternTA

# Install the dependencies
pip install -r requirements.txt

# Set API access token (optional)
# Create or edit the .env file in the project root directory, add API_TOKEN=your-secret-token

# Start demo (The default port is 8080. You can change it if necessary)
sh run.sh

# View run logs 
tail -f nohup.out
```

## API Authentication

The InternTA API server supports authentication using Bearer tokens. To enable this feature:

1. Set the `API_TOKEN` environment variable in the `.env` file in the project root directory:
   ```
   API_TOKEN=your-secret-token
   ```

2. Include the Authorization header in your requests to the API:
   ```
   Authorization: Bearer your-secret-token
   ```

3. If `API_TOKEN` is not set in the `.env` file, authentication will be skipped, and the API will allow all requests.

4. You can test the authentication feature using the provided `test_auth.py` script:
   ```sh
   python test_auth.py
   ```

## User Guide

### 1. Dataset Agent Training

Install dependencies.

```sh
pip install -r requirements.txt
```

Generate high-quality training dataset.

```sh
cd data
python generate_data.py
```

### 2. Training Agent Fine-Tuning

Go to the project root directory

```sh
cd $ROOT_PATH 
```

Check if there is a file named `personal_assistant.json` in the data directory.

```sh
ls -lh data
```

Fine-tune the model using data generated by the Dataset Agent and the Xtuner tool.

```sh
sh train.sh
```

Observe the model weights in the train directory. The naming convention for the directory is `pth_$NUM_EPOCH`.
```sh
ls -lh train
```

Merge the fine-tuned Adapter into the base model.

```sh
# Note: You need to pass the suffix of the directory containing the weights to be merged as a parameter to specify which LORA parameters to merge.
sh merge.sh $NUM_EPOCH
```

### 3. Local Model Evaluation

Test the final merged model in the final directory.

```sh
# Note: Modify the model path as needed
sh chat.sh
```

### 4. RAG Agent Evaluation
 
This section is used to calculate the ROUGE similarity scores for responses generated by the InternTA model and generate evaluation results.

```sh
# Ensure your SynBio-Bench.json file is in the correct directory
pytest ./test/test_model_evaluation.py
```

This command will process the data file and output the results to the `test_results.csv` file.

## Special Thanks

- [DeepSeekR1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [internDog](https://github.com/BestAnHongjun/InternDog)
- [Peft](hhttps://github.com/huggingface/peft)
