# Three Program Versions of Masked Language Model 

The following three versions are almost similar to the online tutorial in the [Huggingface website](https://huggingface.co/learn/nlp-course/en/chapter7/3?fw=pt).

## Version 1

[CODE/train_mlm_v1_ok_multi_gpu.py](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v1_ok_multi_gpu.py)
[CODE/log/log_ver1.txt](https://github.com/unose/masked_lang_model/blob/main/CODE/log/log_ver1.txt)
[Demo video](https://unomail-my.sharepoint.com/:v:/g/personal/myoungkyu_unomaha_edu/Edoy9g3mBkhEsKIaZYJoJYkBbgWvkt6PqJlc6nBze8EayQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rFeeHY)

This version of the code utilizes multiple GPUs by leveraging the `notebook_launcher()` function from the `accelerate` library. The `training_function()` implements the training and evaluation loop, performing the following steps:

***In training loop***, the model is set to training mode with `model.train()`. The training loop iterates through the batches of data from `train_dataloader`. The model processes the batch to produce `outputs`. The loss is computed from the model's outputs.
The backward pass is performed with `accelerator.backward(loss)`.
The optimizer updates the model's parameters.
The learning rate scheduler steps to adjust the learning rate.
The optimizer's gradients are zeroed out for the next iteration.
A progress bar is updated to track training progress.

***In evaluation loop***, the model is set to evaluation mode with `model.eval()`.
The evaluation loop iterates through the batches of data from `eval_dataloader`.
For each batch, the model processes the batch without computing gradients (using `torch.no_grad()`).
The loss for each batch is computed and gathered across GPUs.
The average loss across all batches is computed.
The average loss for the current epoch is logged.

***The perplexity metric***, commonly used for evaluating language models, is computed from the average loss.
The perplexity for the current epoch is logged.
The probabilities are computed from the gathered losses using the softmax function.
The entropy is calculated from these probabilities, providing a measure of uncertainty in the model's predictions.
The entropy for the current epoch is logged.

The model's state is saved at the end of each epoch.
The `accelerator.wait_for_everyone()` ensures that all processes reach this point before proceeding.
The model and tokenizer are saved using the `accelerator.save` function and the `save_pretrained` method.

By utilizing multiple GPUs, this implementation aims to improve the efficiency and speed of the training process. The `notebook_launcher` function facilitates the distribution of the training process across the specified number of GPUs (`num_processes = args.ngpu`). The code also includes periodic checks on GPU usage to monitor resource utilization.


## Version 2

[CODE/train_mlm_v2_nok_multi_gpu.py](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v2_nok_multi_gpu.py)

This version does not use the `notebook_launcher` function. Instead, the `CUDA_VISIBLE_DEVICES` environment variable is set to "0,1" in the shell command. In addition, the Python script uses os.environ["CUDA_VISIBLE_DEVICES"] = "0,1". This ensures that only the specified GPUs (GPU 0 and GPU 1) are visible to the script.

```
CUDA_VISIBLE_DEVICES=0,1 python train_mlm_v2_nok_multi_gpu.py -train 1000 -test 100 -ngpu 2 -epoch 3 -logfile log_v2.txt
```

***However***, the [demo video](https://unomail-my.sharepoint.com/:v:/g/personal/myoungkyu_unomaha_edu/EZASbkSzTTJImGI_nsfEwT4B8kH358eeYl_8QE_e6jM0-g?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=nTaAkG) shows that this version is unable to use multiple GPUs.  See the log file [CODE/log/log_ver2.txt](https://github.com/unose/masked_lang_model/blob/main/CODE/log/log_ver2.txt).


## Version 3

[CODE/train_mlm_v3_err_multi_gpu.py](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v3_err_multi_gpu.py)

This version of the code utilizes the `notebook_launcher()` function, which unexpectedly results in a runtime error. Upon comparing it with the previously functional version, it became evident that the line responsible for the error is `torch.cuda.is_available()` (refer to [line 74](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v3_err_multi_gpu.py#L74)). This function call checks whether CUDA-enabled GPUs are available for use. 

To diagnose the issue further, additional investigation into why `torch.cuda.is_available()` fails is necessary. 