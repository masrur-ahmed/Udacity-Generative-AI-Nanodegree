# Lightweight Fine-Tuning Project


* PEFT technique: LoRA on quantized model
* Model: Gemma-7B
* Evaluation approach: accuracy_score from sklearn on sentiment keyword matching
* Fine-tuning dataset: [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

## Loading and Evaluating a Foundation Model



```python
%%capture
!pip install -Uqqq pip --progress-bar off
!pip install -qqq langchain==0.0.228 --progress-bar off
!pip install -qqq chromadb==0.3.26 --progress-bar off
!pip install -qqq sentence-transformers==2.2.2 --progress-bar off
!pip install -qqq auto-gptq==0.2.2 --progress-bar off
!pip install -qqq einops==0.6.1 --progress-bar off
!pip install -qqq unstructured==0.8.0 --progress-bar off
!pip install -qqq torch==2.0.1 --progress-bar off
```


```python
%%capture
!pip install datasets==2.16.0
!pip install -qqq bitsandbytes
!pip install -qqq  git+https://github.com/lyhue1991/torchkeras 
!pip install -qqq git+https://github.com/lvwerra/trl.git
!pip install -q transformers=="4.38.2"
!pip install -Uqqq git+https://github.com/huggingface/peft  
```


```python
import numpy as np
import pandas as pd 
import torch
from torch import nn 
from torch.utils.data import DataLoader 

import warnings 
warnings.filterwarnings('ignore')

import accelerate 
import peft 

from transformers import AutoTokenizer, AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['XLA_USE_BF16'] = "1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

import torchkeras

```

    2024-03-30 17:36:02.668307: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-03-30 17:36:02.668424: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-03-30 17:36:02.795926: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered



```python
from huggingface_hub import notebook_login
notebook_login()

```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
from transformers import AutoTokenizer, AutoModelForCausalLM

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    llm_int8_has_fp16_weight=False,
        
)



max_seq_length = 2048
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", max_seq_length=max_seq_length)
EOS_TOKEN = tokenizer.eos_token

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", quantization_config=bnb_config)
model.config.use_cache = False
model.config.pretraining_tp = 1

input_text = "Write me about roronoa zoro from one piece"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

```


    tokenizer_config.json:   0%|          | 0.00/1.11k [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/4.24M [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/17.5M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/555 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/629 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/20.9k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/4 [00:00<?, ?it/s]



    model-00001-of-00004.safetensors:   0%|          | 0.00/5.00G [00:00<?, ?B/s]



    model-00002-of-00004.safetensors:   0%|          | 0.00/4.98G [00:00<?, ?B/s]



    model-00003-of-00004.safetensors:   0%|          | 0.00/4.98G [00:00<?, ?B/s]



    model-00004-of-00004.safetensors:   0%|          | 0.00/2.11G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/137 [00:00<?, ?B/s]


    <bos>Write me about roronoa zoro from one piece
    
    Answer:
    
    Step 1/



```python
df_train=pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_training.csv')
df_val=pd.read_csv('/kaggle/input/twitter-entity-sentiment-analysis/twitter_validation.csv')
```


```python
df_train.columns
```




    Index(['2401', 'Borderlands', 'Positive',
           'im getting on borderlands and i will murder you all ,'],
          dtype='object')




```python
df_val.columns
```




    Index(['3364', 'Facebook', 'Irrelevant',
           'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣'],
          dtype='object')




```python

df_train.rename(columns={'Positive': 'sentiment','im getting on borderlands and i will murder you all ,':'text' }, inplace=True)
df_val.rename(columns={'Irrelevant': 'sentiment','I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣':'text' }, inplace=True)
df_val=df_val.drop(['3364','Facebook'],axis=1)
df_train=df_train.drop(['Borderlands','2401'],axis=1)
```


```python
from sklearn.model_selection import train_test_split

df_train = df_train[df_train['sentiment'] != 'Irrelevant']
df_val = df_val[df_val['sentiment'] != 'Irrelevant']

df_train['sentiment'] = df_train['sentiment'].str.lower()
df_val['sentiment'] = df_val['sentiment'].str.lower()

X_train = list()
X_test = list()
columns_to_check = ['positive', 'negative', 'neutral']
for sentiment in columns_to_check:
    train, test = train_test_split(df_train[df_train.sentiment == sentiment], 
                                   train_size=200,
                                   test_size=50, 
                                   random_state=42)
    X_train.append(train)
    X_test.append(test)

X_train = pd.concat(X_train).sample(frac=1, random_state=10)
X_test = pd.concat(X_test)

eval_idx = [idx for idx in df_train.index if idx not in list(train.index) + list(test.index)]
X_eval = df_train[df_train.index.isin(eval_idx)]
X_eval = (X_eval
          .groupby('sentiment', group_keys=False)
          .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
X_train = X_train.reset_index(drop=True)
from langchain.prompts import PromptTemplate


def generate_prompt(data_point):
    return f"""generate_prompt
            Analyze the sentiment of the comment enclosed in square brackets, 
            determine if it is vulgar, hate, religious, threat, troll, insult and neutral. It can have multiple labels among them. return the answer as 
            the corresponding sentiment label vulgar, hate, religious, threat, troll, insult or neutral. 

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip() + EOS_TOKEN

def generate_test_prompt(data_point):
    return f"""generate_prompt
            Analyze the sentiment of the comment enclosed in square brackets, 
            determine if it is positive, negative or neutral. It can have multiple labels among them. return the answer as 
            the corresponding sentiment label positive, negative or neutral. 


            [{data_point["text"]}] = 

            """.strip()

X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), 
                       columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), 
                      columns=["text"])

y_true = X_test.sentiment
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])


```


```python
from datasets import Dataset

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)
```


```python
X_train['text'][41]
```




    'generate_prompt\n            Analyze the sentiment of the comment enclosed in square brackets, \n            determine if it is vulgar, hate, religious, threat, troll, insult and neutral. It can have multiple labels among them. return the answer as \n            the corresponding sentiment label vulgar, hate, religious, threat, troll, insult or neutral. \n\n            [The Johnson & General Johnson Halts Talc Ltd Baby Powder Horse Sales Management wb. md / 2ylv9sV] = neutral<eos>'




```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred):
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)
    
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Overall Accuracy: {accuracy:.3f}')
    
    unique_labels = set(y_true_mapped)
    
    for label in unique_labels:
        label_indices = [i for i, y in enumerate(y_true_mapped) if y == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')
        
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels)
    print('\nClassification Report:')
    print(class_report)
    
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

```


```python
def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1, temperature=0.0)
        result = tokenizer.decode(outputs[0])
        answer = result.split("=")[-1].lower()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
    return y_pred

```


```python
from tqdm import tqdm
y_pred = predict(X_test, model, tokenizer)
```

      0%|          | 0/150 [00:00<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
    100%|██████████| 150/150 [02:51<00:00,  1.14s/it]



```python
evaluate(y_true, y_pred)
```

    Overall Accuracy: 0.587
    Accuracy for label positive: 0.840
    Accuracy for label neutral: 0.020
    Accuracy for label negative: 0.900
    
    Classification Report:
                  precision    recall  f1-score   support
    
        positive       0.64      0.84      0.72        50
         neutral       1.00      0.02      0.04        50
        negative       0.54      0.90      0.68        50
    
        accuracy                           0.59       150
       macro avg       0.73      0.59      0.48       150
    weighted avg       0.73      0.59      0.48       150
    
    
    Confusion Matrix:
    [[42  0  8]
     [19  1 30]
     [ 5  0 45]]


The base model achieves an accuracy of 58.7%. Additionally, it performs poorly on negative texts but excels with neutral ones.

## Performing Parameter-Efficient Fine-Tuning



```python
from peft import get_peft_config, get_peft_model, TaskType

# Enable gradient checkpointing support in the model
model.supports_gradient_checkpointing = True  #

# Enable gradient checkpointing for more memory-efficient training
model.gradient_checkpointing_enable()

# Enable requiring gradients for model inputs
model.enable_input_require_grads()

# Disable cache usage in the model configuration to silence warnings 
model.config.use_cache = False  #  Re-enable for inference
```


```python
import bitsandbytes as bnb

def find_all_linear_names(model):
    """
    Find all fully connected layers and add low-rank adapters to each one.
    """
    cls = bnb.nn.Linear4bit

    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

lora_modules = find_all_linear_names(model)
print(lora_modules)
```

    ['base_layer']



```python
from peft import LoraConfig

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=lora_modules
)

peft_model = get_peft_model(model, peft_config)

peft_model.is_parallelizable = True
peft_model.model_parallel = True



```


```python
for name, para in peft_model.named_parameters():
    # Break the loop if the name contains '.1.'
    if '.1.' in name:
        break
    # Check if the parameter is related to LoRA (contains 'lora' in its name)
    if 'lora' in name.lower():
        # Print information about the parameter
        print(name + ':')
        print('shape = ', list(para.shape), '\t', 'sum = ', para.sum().item())
        print('\n')
```

    base_model.model.model.layers.0.self_attn.q_proj.base_layer.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  1.22685968875885
    
    
    base_model.model.model.layers.0.self_attn.q_proj.base_layer.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -1.334961175918579
    
    
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  -0.005172288045287132
    
    
    base_model.model.model.layers.0.self_attn.k_proj.base_layer.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  0.9425515532493591
    
    
    base_model.model.model.layers.0.self_attn.k_proj.base_layer.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  0.5233312845230103
    
    
    base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  -0.025704532861709595
    
    
    base_model.model.model.layers.0.self_attn.v_proj.base_layer.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -1.265538215637207
    
    
    base_model.model.model.layers.0.self_attn.v_proj.base_layer.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  0.1325361132621765
    
    
    base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight:
    shape =  [4096, 8] 	 sum =  -0.0974748432636261
    
    
    base_model.model.model.layers.0.self_attn.o_proj.base_layer.lora_A.default.weight:
    shape =  [8, 4096] 	 sum =  -0.08451735973358154
    
    
    base_model.model.model.layers.0.self_attn.o_proj.base_layer.lora_B.default.weight:
    shape =  [3072, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight:
    shape =  [8, 4096] 	 sum =  1.963411808013916
    
    
    base_model.model.model.layers.0.self_attn.o_proj.lora_B.default.weight:
    shape =  [3072, 8] 	 sum =  0.03276874125003815
    
    
    base_model.model.model.layers.0.mlp.gate_proj.base_layer.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -0.4273686408996582
    
    
    base_model.model.model.layers.0.mlp.gate_proj.base_layer.lora_B.default.weight:
    shape =  [24576, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -1.2665225267410278
    
    
    base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight:
    shape =  [24576, 8] 	 sum =  0.0793817937374115
    
    
    base_model.model.model.layers.0.mlp.up_proj.base_layer.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -5.348316192626953
    
    
    base_model.model.model.layers.0.mlp.up_proj.base_layer.lora_B.default.weight:
    shape =  [24576, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight:
    shape =  [8, 3072] 	 sum =  -1.1400129795074463
    
    
    base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight:
    shape =  [24576, 8] 	 sum =  -0.005167707800865173
    
    
    base_model.model.model.layers.0.mlp.down_proj.base_layer.lora_A.default.weight:
    shape =  [8, 24576] 	 sum =  -2.2288260459899902
    
    
    base_model.model.model.layers.0.mlp.down_proj.base_layer.lora_B.default.weight:
    shape =  [3072, 8] 	 sum =  0.0
    
    
    base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight:
    shape =  [8, 24576] 	 sum =  -1.0751280784606934
    
    
    base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight:
    shape =  [3072, 8] 	 sum =  0.01632239855825901
    
    



```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./Gemma-7b",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=25,
    logging_strategy="steps",
    max_steps=-1,
    optim="paged_adamw_32bit",
    save_steps=0,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    weight_decay=0.001,
    num_train_epochs=3,
    warmup_ratio=0.03,
    group_by_length=False,
    evaluation_strategy='steps',
    eval_steps=112,
    eval_accumulation_steps=1,
    lr_scheduler_type="cosine"
)

```


```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,
)
```


    Map:   0%|          | 0/600 [00:00<?, ? examples/s]



    Map:   0%|          | 0/150 [00:00<?, ? examples/s]



```python
trainer.train()


trainer.model.save_pretrained("trained-model")
```



    <div>

      <progress value='225' max='225' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [225/225 1:23:48, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>112</td>
      <td>0.735000</td>
      <td>4.438314</td>
    </tr>
    <tr>
      <td>224</td>
      <td>0.288800</td>
      <td>4.996609</td>
    </tr>
  </tbody>
</table><p>


## Performing Inference with a PEFT Model



```python
model_trained = model

```


```python
from peft import PeftModel
ft_model = PeftModel.from_pretrained(model_trained, "/kaggle/working/trained-model",torch_dtype=torch.float16,is_trainable=False)
```


```python
from tqdm import tqdm
y_pred = predict(X_test, ft_model, tokenizer)
```

    100%|██████████| 150/150 [03:08<00:00,  1.25s/it]



```python
evaluate(y_true, y_pred)
```

    Overall Accuracy: 0.727
    Accuracy for label positive: 0.860
    Accuracy for label neutral: 0.440
    Accuracy for label negative: 0.880
    
    Classification Report:
                  precision    recall  f1-score   support
    
        positive       0.72      0.86      0.78        50
         neutral       0.76      0.44      0.56        50
        negative       0.72      0.88      0.79        50
    
        accuracy                           0.73       150
       macro avg       0.73      0.73      0.71       150
    weighted avg       0.73      0.73      0.71       150
    
    
    Confusion Matrix:
    [[43  4  3]
     [14 22 14]
     [ 3  3 44]]


While the model's overall accuracy has significantly improved to 72.7%, its accuracy for neutral comments has decreased notably.


```python

```
