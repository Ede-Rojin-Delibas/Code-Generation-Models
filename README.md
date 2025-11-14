### Project title: Fine-tuning a Code LLM on curated permissive GitHub code

#### Short Description: 
A research-style project to build a curated permissive-code dataset and fine-tune an open-source code language model (e.g., StarCoder, CodeLlama) using parameter-efficient methods (PEFT/LoRA). The repo includes data ingestion, preprocessing, fine-tuning notebooks, evaluation harness (execution-based tests), and a Streamlit demo.

__Scope (Week 0–1)__

* Prepare environment and tooling

* Decide compute and base model

* Research PEFT / LoRA, datasets, and GitHub data sources

* Create initial project plan and notebook skeletons

#### Ethics & license note:
This project will use only permissively licensed repositories (MIT/Apache/BSD) for training. All dataset items will include provenance metadata (repo, path, commit SHA, license). The model and dataset cards will document limitations and license filtering.