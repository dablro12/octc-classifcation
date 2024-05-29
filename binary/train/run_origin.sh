python run_origin.py --model "swin" \
                --version "origin" \
                --cuda "0"\
                --ts_batch_size 20\
                --vs_batch_size 8\
                --epochs 51\
                --loss "bce"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --save_path "/home/eiden/eiden/octc-classification/models/binary"\
                --pretrain "no" --pretrained_model "practice" --error_signal "yes"\
                --wandb "yes"\
                > output_origin.log 2>&1 &
