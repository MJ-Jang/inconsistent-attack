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

#### 4) Use generated variable part to make the model produce inconsistent explanations
- Save the final generated output as *inconsist-final-test.json* and locate it in the same directory where *inconsist-expls-test.json* is located
- File format of *inconsist-final-test.json* should be
    - pair_id: key index for mapping
    - context: context part (e.g. premise for NLI)
    - variable: generated inconsistent variable part
    - inconsist_expl: generated explanations with the inconsitent variables
    - label: predicted labels
    ```json
     {
        "pair_id": [...],
        "context": [...],
        "variable": [...],
        "inconsist_expl": [...],
        'label': [...]
     }


#### 5) Extract correct inconsistent explanations
- The original test dataset is required to get the original label
- See *test.tsv* file in **esnli_sample** folder to use the code without modification
- If you used different format for the test data, modify the line 97-101 in *extract_inconsistent_expl.py* file

