import transformers
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(level=logging.DEBUG)

class QADataGen:
    def __init__(self, qa_file, model="distilbert-base-uncased", batch_size=32):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.batch_size = batch_size
        self.n_max_size = 100
        self.n_beam = 30
        self.n_finetune_questions = 2000
        self.n_ans = 2000
        
        # Load QA DataFrame
        self.qa_df = pd.read_csv(qa_file, nrows=self.n_ans)
        self.qa_df.to_csv(f"/qa_{self.n_ans}.csv", index=False)
        self.dataset_final = load_dataset("csv", data_files=f"/qa_{self.n_ans}.csv")
        
        # Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 384  # The maximum length of a feature (question and context)
        self.doc_stride = 8  # The authorized overlap

        # Ensure tokenizer is of the correct type
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

        # Fine-tune the model
        self.finetune_data()

    def finetune_data(self):
        # Load the fine-tuning dataset
        self.dataset_finetune = load_dataset("squad", split=f"train[0:{self.n_finetune_questions}]")
        
        # Prepare training features
        tokenized_datasets = self.dataset_finetune.map(self.prepare_train_features,
                                                       batched=True,
                                                       remove_columns=self.dataset_finetune.column_names)
        
        model_name = self.model.split("/")[-1]
        args = TrainingArguments(
            f"{model_name}-finetuned-squad",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            report_to="none",
        )

        data_collator = default_data_collator
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model)
        self.trainer = Trainer(
            self.qa_model,
            args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def prepare_train_features(self, examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]

        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Adding start/end positions
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        examples["question"] = [str(q).lstrip() for q in examples["question"]]
        examples["context"] = [str(c) for c in examples["context"]]

        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def post_process(self, raw_predictions, validation_features2):
        start_logits = raw_predictions.predictions[0]
        end_logits = raw_predictions.predictions[1]
        start_indexes = np.argsort(start_logits, axis=-1)[:, -1: -self.n_beam - 1: -1]
        end_indexes = np.argsort(end_logits, axis=-1)[:, -1: -self.n_beam - 1: -1]
        valid_answers_list = []
        offset_mapping = validation_features2["offset_mapping"]
        example_ids = validation_features2['example_id']
        context = self.dataset_final["train"]["context"]
        input_ids = validation_features2['input_ids']

        for row in range(start_indexes.shape[0]):
            valid_answers = []
            example_id = example_ids[row]
            for col_start in range(start_indexes.shape[1]):
                for col_end in range(end_indexes.shape[1]):
                    start_index = start_indexes[row, col_start]
                    end_index = end_indexes[row, col_end]
                    if (start_index < end_index) and (start_index < start_logits.shape[1]) and (
                            end_index < end_logits.shape[1]) and (end_index - start_index + 1 <= self.n_max_size):
                        if offset_mapping[row] and offset_mapping[row][start_index] and offset_mapping[row][end_index]:
                            start_char = offset_mapping[row][start_index][0]
                            end_char = offset_mapping[row][end_index][1]
                            context_q = context[example_ids[row]]
                            if end_char < len(context_q):
                                text = context_q[start_char:end_char + 1]
                                text = text.strip()
                                if len(text):
                                    valid_answers.append(
                                        {
                                            "score": start_logits[row, start_index] + end_logits[row, end_index],
                                            "text": text
                                        }
                                    )
            valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
            if valid_answers:
                valid_answers[0]['qid'] = example_id
                valid_answers_list.append(valid_answers[0])
            else:
                valid_answers_list.append({'score': -1, 'text': '', 'qid': example_id})

        return valid_answers_list

    def finetune(self, save_model=None):
        train_losses = []
        eval_losses = []

        # Custom callback to log losses
        class LossLogger(transformers.TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                last_log = state.log_history[-1] if state.log_history else {}
                if "loss" in last_log:
                    train_losses.append(last_log["loss"])
                if "eval_loss" in last_log:
                    eval_losses.append(last_log["eval_loss"])

        self.trainer.add_callback(LossLogger())
        self.trainer.train()
        
        if save_model:
            self.trainer.save_model("test-squad-trained")
        self.logger.info("Fine-tuned underlying QA model")

        # Plot losses
        self.plot_losses(train_losses, eval_losses)

    def plot_losses(self, train_losses, eval_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(eval_losses, label='Evaluation Loss')
        plt.title('Model Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def answer_questions(self):
        validation_features2 = self.dataset_final["train"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.dataset_final["train"].column_names
        )
        raw_predictions = self.trainer.predict(validation_features2)
        validation_features2.set_format(type=validation_features2.format["type"],
                                        columns=list(validation_features2.features.keys()))
        valid_answers_list = self.post_process(raw_predictions, validation_features2)
        answers = [v['text'] for v in valid_answers_list]
        qid = [v['qid'] for v in valid_answers_list]
        scores = [v['score'] for v in valid_answers_list]
        df_ans = pd.DataFrame(data={'question': list(range(len(valid_answers_list))), 'answer': answers, 'qid': qid,
                                    'score': scores})
        return df_ans

    def cb_finetune_file(self, save_ans_file):
        self.finetune()
        df = self.answer_questions()
        df2 = df[["qid", "score"]].groupby(["qid"]).max().reset_index(drop=False)
        df2 = pd.merge(df2, df[["qid", "score", "answer"]], on=["qid", "score"], how="left")
        df2 = df2.groupby(["qid"]).first().reset_index(drop=False)
        qa_df = self.qa_df.drop(columns=["qid"])
        qa_df.rename(columns={"id": "qid"}, inplace=True)
        df2 = pd.merge(df2, self.qa_df[["id", "question"]], on=["qid"], how="inner")
        df2.to_csv(save_ans_file, index=False)

if __name__ == "__main__":
    qa_file = "/final_short.csv"  # Path to your QA dataset CSV file
    savefile = "/answers_qa.csv"   # Path to save answers
    qa = QADataGen(qa_file=qa_file)
    qa.cb_finetune_file(save_ans_file=savefile)