python run.py --model "convnext" \
                --version "ours" \
                --cuda "0"\
                --ts_batch_size 80\
                --vs_batch_size 8\
                --epochs 100\
                --loss "bce"\
                --optimizer "Adam"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --save_path "/home/eiden/eiden/octc-classification/models/binary"\
                --pretrain "no" --pretrained_model "practice" --error_signal "yes"\
                --wandb "yes"\
                > output.log 2>&1 &
