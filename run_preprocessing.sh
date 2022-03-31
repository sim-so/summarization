DIR=./data/
TRAIN_FN=train.tsv
TEST_FN=test.tsv

shuf {DIR}{TRAIN_FN} > {DIR}full_train.shuf.tsv
tail -n 500 {DIR}full_train.shuf.tsv > {DIR}valid.shuf.tsv
head -n 2494 {DIR}full_train.shuf.tsv > {DIR}.train.shuf.tsv