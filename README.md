# inconsistent-attack

### Installation
- Please install required package in requirements.txt
- Download conceptnet transformed file
    ```bash
    bash download_conceptnet_en.sh
    ```

### Steps
#### 1) Train reverse explainer
- Construct a dataset for training reverse explainer
- We used T5-base sturcture. The dataset is a *tsv* file that has three colums: pairID, input, label
- Both input and label has a free text form where sentence type tags are added
- The sentence type tag of explanation is **Explanation:**
- Please refer the sample data provided in *resources/esnli_sample* folder
- Usage example)
    ```bash
    cd src
    bash train_rev_explainer.sh 0 ../resources/esnli_sample/ ../model_binary/
    ```
    
#### 2) Generate inconsistent explanations
- Output file format: json
    - pair_id: key index for mapping
    - context: context part (e.g. premise for NLI)
    - inconsist_expl: generated inconsistent explanations
    - tags: type of method used for generating inconsistent explanations
    ```json
     {
        "pair_id": [...],
        "context": [...],
        "inconsist_expl": [...],
        'tags': [...]
     }
    ``` 
- Usage example)
    ```bash
    cd src
    python generate_inconsistent_expl.py --data_dir ../resources/esnli_sample/ --save_dir ../resources/esnli_sample/
    ```

#### 3) Generate inconsistent variable parts by using ReverseExplainer
- Output file format: json
    - pair_id: key index for mapping
    - context: context part (e.g. premise for NLI)
    - variable: generated inconsistent variable part
    - tags: type of method used for generating inconsistent explanations
    ```json
     {
        "pair_id": [...],
        "context": [...],
        "variable": [...],
        'tags': [...]
     }
    ``` 
- Usage example)
    ```bash
    cd src
    bash generate_inconsistent_var.sh ../resources/esnli_sample/ ../resources/esnli_sample/
    ```
