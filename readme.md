### this is quick note

I conducted a small experiment. Overall, I used 150 samples from the MATH-500 dataset as a calibration set, aiming to identify a meaningful quantile.

Some quick look about this dataset

Experiment Setting
| type | Reasoning Model                    | Judge Model                  | Pred_length Model            |
|------|------------------------------------|------------------------------|------------------------------|
| Model| DeepSeek-R1-Distill-Qwen-32B       | Qwen/Qwen2.5-14B-Instruct    | Qwen/Qwen2.5-14B-Instruct    |



To simplify the process, I'm currently search by 1%, 2%, ... 100% instead of token by token.(however,the code has been advanced to use binary search, so we can search token by token and the complexity would not increas significantly)

here the min_valid_length is the minist k that using the firsk k reasonging process can llm output the same answer as the answer after full reasoning.

the min_valid_percent is  $\frac{min\_ valid\_ length}{full\_reasoning\_length}\cdot 100$


|       index       | min_valid_length | min_valid_percent |
|------------------|------------------|-------------------|
| mean             | 3028.476510      | 43.550336         |
| std              | 6404.572198      | 35.975486         |
| min              | 4.000000         | 1.000000          |
| 25%              | 106.000000       | 13.000000         |
| 50%              | 603.000000       | 35.000000         |
| 75%              | 1638.000000      | 79.000000         |
| max              | 31147.000000     | 100.000000        |


the relationship between min valid length and min valid percent
![relationship between min valid length and min valid percent](minlength-minpercent.png)

the relationship between min valid length and pred length
![relationship between min valid length and pred length](minlength-predlength.png)

- the quantile

    | $\alpha$ | $\hat{\tau}$ |
    |----------|--------------|
    | 0.1      | 12098.033557047002 |
    | 0.2      | 2433.9463087248396 |
    | 0.3      | 1360.5906040268455 |



### Some Note

1. The reasons why the min_valid_percent is 100 maybe
    - the full reasoning process is too long so the result maybe None (because it more than the defined max token)  (mainly)
    - the answer after the full reasoning process is wrong, but the answer by judge LLM is correct (a few)
    - I simply use == or in to check the answer, it useful in almost math problem, but in some case the form of output may different, maybe we can use a simple llm to judge whether the output is same, but in this case I jump this process.
    - APIError. Because I 

2. The prediction of Length is too conservative and wierd, I may adjust the prompt and calibrate more


### Future Work

