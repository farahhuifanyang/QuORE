This is the official code repository for "NAME_OF_PAPER" ([preprint]()) by [Huifan Yang](mailto:huifunny@bupt.edu.cn), [Da-Wei Li](mailto:daweilee@microsoft.com) and [Bin Wu](mailto:wubin@bupt.edu.cn).  



## Usage
The commands listed in this section need to be run from the root directory of the repository.

First, install prerequisites with  
```pip install -r requirements.txt```

### Commands
* Train a model:  
```allennlp train configs/[config_file] -s [training_directory] --include-package src```

* Output predictions by a model:  
```allennlp predict [output_directory]/model.tar.gz [test_file] --predictor machine-comprehension --cuda-device 0 --output-file [output_directory]/predictions.jsonl --use-dataset-reader --include-package src```

* Evaluate a model:  
```allennlp evaluate [output_directory]/model.tar.gz [test_file] --cuda-device 0 --output-file [output_directory]/eval.json --include-package src```
