python ./run.py --model "swin" \
                --version "ours" \
                --cuda "0"\
                --ts_batch_size 48\
                --vs_batch_size 8\
                --epochs 200\
                --loss "ce"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --save_path "/home/eiden/eiden/octc-classification/models/multi"\
                --pretrain "no" --pretrained_model "practice" --error_signal no\
                --wandb "yes"\
                > output.log 2>&1 &