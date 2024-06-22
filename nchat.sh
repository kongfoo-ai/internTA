xtuner convert pth_to_hf train/internlm2_1_8b_qlora_alpaca_e3_copy.py train/iter_$1.pth huggingface
xtuner convert merge model huggingface final
xtuner chat model --adapter huggingface --prompt-template internlm2_chat