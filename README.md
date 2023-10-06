# FLARE23

This is a solution for FLARE23 

First you should process all the data by `flare_process.py`

```
In the flare_process.py you should modify the data path variable such as:FLARE_path, FLARE_label_path, flare_to_path ,flare_label_to_path, which means the image data path, and the label path, after process image save path, and after process label save path.
```



Second you should generate a .json file, the format of the json file you can  refer to our publicly available json file.

```
You need to put all the paths to the corresponding images and labels in a dict, then store all the dicts in a list of FLARE_val and finally save it as a json file. The exact format can be found in our .json file. 
```

You can also used the `save_json` function to generate the json file in the `flare_process.py`

```
 the flare_path is the after process the image root, and the flare_label_path is the label after process root.
```



Third we open source our fine-tune model it is best_model.pth in the repository. 

the fine-tune model can get by link：https://pan.baidu.com/s/1fbhMcpwA81SYcxRWbKn7sA 
password：4egx 



Finally, you can run the `FLARE_Test.py` to test the validation mertic.

In this step, you should modify the json file path variable:`data_path`, and the model path variable：`model_path`.

