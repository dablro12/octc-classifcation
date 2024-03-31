python run.py --model "unet" \
                --version "sonography" \
                --cuda "0"\
                --ts_batch_size 32\
                --vs_batch_size 8\
                --epochs 500\
                --loss "bce"\
                --optimizer "Adam"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --pretrain "yes" --pretrained_model "unet_sonography_240124" --error_signal "yes"\
                --wandb "yes"\
                # --training_date "230124"
