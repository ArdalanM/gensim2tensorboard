# gensim2tensorboard
Train word embeddings with gensim and visualize them with TensorBoard.
![fig](fig/fig.png "fig")

## Requirements:
- regex
- gensim
- tensorflow (>= 0.12)


## Example:
- Train from text file:
```
python train.py --file "data/SMSSpamCollection.txt" \
                --input_type 'txt' \
                --folder "model_from_txt"
```

- Train from csv file:
```
python train.py --file "data/movie_reviews.csv" \
                --input_type "csv" \
                --separator "," \
                --folder "model_from_csv" \
                --columns_to_select "Phrase"
```

- Train with custom word2vec parameters:
```
python train.py --file "data/movie_reviews.csv" \
                --input_type "csv" \
                --separator "," \
                --folder "model_from_csv_custom_params" \
                --columns_to_select "Phrase" \
                --size 50 \
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
```

Eventially, visualize the embeddings with tensorboard:
```
tensorboard --logdir=./ --reload_interval 1
```

You should be able to run the above script on your own dataset, just
dump them into the `data` folder 