# Three Program Versions of Masked Language Model 

The following three versions are almost similar to the online tutorial in the [Huggingface website](https://huggingface.co/learn/nlp-course/en/chapter7/3?fw=pt).

## Version 1

[CODE/train_mlm_v1_ok_multi_gpu.py](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v1_ok_multi_gpu.py)

This version of the code utilizes multiple GPUs by leveraging the `notebook_launcher` function from the `accelerate` library. The `training_function()` implements the training and evaluation loop, performing the following steps:

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




## Version 3

[CODE/train_mlm_v3_err_multi_gpu.py](https://github.com/unose/masked_lang_model/blob/main/CODE/train_mlm_v3_err_multi_gpu.py)



