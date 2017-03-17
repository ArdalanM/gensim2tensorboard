#!/usr/bin/env bash

# Train from textfile
python3 -m src.train --file "data/SMSSpamCollection.txt" \
                     --input_type 'txt' \
                     --folder "models/SMSSpamCollection" \
                     --size 10 \
                     --alpha 0.025 \
                     --window 5 \
                     --min_count 5 \
                     --max_vocab_size 100000 \
                     --sample 1e-3 \
                     --seed 1 \
                     --workers 4 \
                     --min_alpha 0.0001 \
                     --sg 0 \
                     --hs 0 \
                     --negative 10 \
                     --cbow_mean 1 \
                     --iter 5 \
                     --null_word 0

# Train from csv
python3 -m src.train.py --file "data/movie_reviews.csv" \
                        --input_type "csv" \
                        --separator "," \
                        --folder "model_from_csv_custom_params" \
                        --columns_to_select "Phrase" \
                        --size 10 \
                        --alpha 0.025 \
                        --window 5 \
                        --min_count 5 \
                        --max_vocab_size 100000 \
                        --sample 1e-3 \
                        --seed 1 \
                        --workers 4 \
                        --min_alpha 0.0001 \
                        --sg 0 \
                        --hs 0 \
                        --negative 10 \
                        --cbow_mean 1 \
                        --iter 5 \
                        --null_word 0