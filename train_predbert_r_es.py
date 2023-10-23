import comet_ml
import logging
from transformers import BertTokenizer, BertConfig
from transformers import PredBertForPreTraining
from transformers import TextDatasetForPredBert, TextDatasetForPredBertSSD
from transformers import DataCollatorForPredBertLanguageModeling, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerPred


run_name = 'PredBERT-R-es'
logging.basicConfig(filename='{}.log'.format(run_name), level=logging.INFO)

model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
config.all_layers = True # top-down in all layers
config.until_layer = 1 # 1 for last layer, 6 for half layers
config.k_size = 2
context_size = 3
window = context_size*config.k_size+context_size-config.k_size # overlap of sentences
config.window_size = window

model = PredBertForPreTraining.from_pretrained(model_name, config=config)

directory = '/user/vgaraujo/corpus/train' # path to corpus, assuming inside a folder named data/ with documents *.txt 
train_dataset = TextDatasetForPredBert(tokenizer, directory, block_size=64, window_size=window)
directory = '/user/vgaraujo/corpus/val'
eval_dataset = TextDatasetForPredBert(tokenizer, directory, block_size=64, window_size=window)

data_collator = DataCollatorForPredBertLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.1, overlap_context=True, context_size=context_size,
)

training_args = TrainingArguments(
    output_dir="./{}".format(run_name),
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,  
    per_device_train_batch_size=7,
    per_device_eval_batch_size=7,
    num_train_epochs=10, # overridden
    save_steps=50000, 
    evaluation_strategy="steps",
    eval_steps=3000,
    fp16=True,
    run_name=run_name,
    prediction_loss_only=True,
    logging_first_step=True,
    logging_steps=1000,
    dataloader_drop_last=True,
    max_steps=1000000,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=2,
)

trainer = TrainerPred(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
