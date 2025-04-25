---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:14281
- loss:TripletLoss
base_model: sentence-transformers/all-distilroberta-v1
widget:
- source_sentence: 42, Central Square, Lynn, Essex County, Massachusetts, 01901
  sentences:
  - 81, Blake Street, Lynn, Essex County, Massachusetts, 01901
  - 567, Summer Street, Lynn, Essex County, Massachusetts, 01905
  - 179, Sydney Street, Unit 2, Dorchester, Suffolk County, Massachusetts, 02125
- source_sentence: 401, Engamore Lane, Unit 208, Norwood, Norfolk County, Massachusetts,
    02062
  sentences:
  - 30, Shipway Place, Unit 30, Charlestown, Suffolk County, Massachusetts, 02129
  - 303, Village Road east, Norwood, Norfolk County, Massachusetts, 02062
  - 15, Sherrill Road, Marshfield, Plymouth County, Massachusetts, 02050
- source_sentence: 64, Rosseter Street, Unit 2, Dorchester, Suffolk County, Massachusetts,
    02121
  sentences:
  - 25, Highland Street, Unit 3, Revere, Suffolk County, Massachusetts, 02151
  - 2, Ellis Street, Unit b, Roxbury, Suffolk County, Massachusetts, 02119
  - 214, Washington Street, Dorchester, Suffolk County, Massachusetts, 02121
- source_sentence: 21, Wendell Street, Unit 11, Cambridge, Middlesex County, Massachusetts,
    02138
  sentences:
  - 5, Wendell Street, Unit 6, Cambridge, Middlesex County, Massachusetts, 02138
  - 27, Captain Bellamy Lane, Centerville, Barnstable County, Massachusetts, 02632
  - 324, Franklin Street, Cambridge, Middlesex County, Massachusetts, 02139
- source_sentence: 179, Thomas Burgin Parkway, Unit 7, Quincy, Norfolk County, Massachusetts,
    02169
  sentences:
  - 176, Presidents Lane, Unit 407, Quincy, Norfolk County, Massachusetts, 02169
  - 75, Station Landing, Unit 110, Medford, Middlesex County, Massachusetts, 02155
  - 179, Thomas Burgin Parkway, Unit 7, Quincy, Norfolk County, Massachusetts, 02169
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on sentence-transformers/all-distilroberta-v1
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all distilroberta v1 val
      type: all-distilroberta-v1-val
    metrics:
    - type: cosine_accuracy
      value: 0.8613736629486084
      name: Cosine Accuracy
---

# SentenceTransformer based on sentence-transformers/all-distilroberta-v1

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1) <!-- at revision 842eaed40bee4d61673a81c92d5689a8fed7a09f -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '179, Thomas Burgin Parkway, Unit 7, Quincy, Norfolk County, Massachusetts, 02169',
    '176, Presidents Lane, Unit 407, Quincy, Norfolk County, Massachusetts, 02169',
    '75, Station Landing, Unit 110, Medford, Middlesex County, Massachusetts, 02155',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `all-distilroberta-v1-val`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.8614** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 14,281 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | sentence_2                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 16 tokens</li><li>mean: 19.9 tokens</li><li>max: 30 tokens</li></ul> | <ul><li>min: 16 tokens</li><li>mean: 19.95 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 19.85 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                   | sentence_1                                                                                  | sentence_2                                                                                   |
  |:---------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
  | <code>78, Jouvette Street, Unit 3, New Bedford, Bristol County, Massachusetts, 02744</code>  | <code>40, Winsor Street, Unit 2, New Bedford, Bristol County, Massachusetts, 02744</code>   | <code>629, Purchase Street, Unit 2, New Bedford, Bristol County, Massachusetts, 02740</code> |
  | <code>99, Seton Highlands, Plymouth, Plymouth County, Massachusetts, 02360</code>            | <code>87, Seton Highlands, Plymouth, Plymouth County, Massachusetts, 02360</code>           | <code>1125, north Main Street, Randolph, Norfolk County, Massachusetts, 02368</code>         |
  | <code>73, Collette Street, Unit 1s, New Bedford, Bristol County, Massachusetts, 02746</code> | <code>62, Hathaway Street, Unit 1, New Bedford, Bristol County, Massachusetts, 02746</code> | <code>204, Rockdale Avenue, Unit 2, New Bedford, Bristol County, Massachusetts, 02740</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.COSINE",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | all-distilroberta-v1-val_cosine_accuracy |
|:------:|:----:|:-------------:|:----------------------------------------:|
| 0.1120 | 100  | -             | 0.8570                                   |
| 0.2240 | 200  | -             | 0.8551                                   |
| 0.3359 | 300  | -             | 0.8444                                   |
| 0.4479 | 400  | -             | 0.8437                                   |
| 0.5599 | 500  | 4.3295        | 0.8601                                   |
| 0.6719 | 600  | -             | 0.8570                                   |
| 0.7839 | 700  | -             | 0.8557                                   |
| 0.8959 | 800  | -             | 0.8557                                   |
| 1.0    | 893  | -             | 0.8570                                   |
| 1.0078 | 900  | -             | 0.8570                                   |
| 1.1198 | 1000 | 4.115         | 0.8601                                   |
| 1.2318 | 1100 | -             | 0.8538                                   |
| 1.3438 | 1200 | -             | 0.8614                                   |


### Framework Versions
- Python: 3.11.12
- Sentence Transformers: 3.4.1
- Transformers: 4.51.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->