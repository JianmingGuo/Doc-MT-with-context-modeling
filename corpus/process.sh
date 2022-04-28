SCRIPTS=/home/ming/tools/mosesdecoder/scripts
TRUECASE=$SCRIPTS/recaser/truecase.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl


lang1=en
lang2=de
bpe_prefix=/home/ming/project/MPE/general/bpe

train_file_prefix='corpus/train'
valid_file_prefix='corpus/valid'
test_file_prefix='corpus/test'

echo "processing  "$train_file_prefix.$lang1 " ... "

python /home/ming/tools/subword-nmt/subword_nmt/learn_bpe.py -s 32000 -t < $train_file_prefix.$lang1 > bpe.$lang1 
python /home/ming/tools/subword-nmt/subword_nmt/apply_bpe.py -c $bpe_prefix.$lang1 < $train_file_prefix.$lang1 > $train_file_prefix.bpe.42k.$lang1


echo "processing  "$train_file_prefix.$lang2 " ... "

python /home/ming/tools/subword-nmt/subword_nmt/learn_bpe.py -s 32000 -t < $train_file_prefix.$lang2 > bpe.$lang2
python /home/ming/tools/subword-nmt/subword_nmt/apply_bpe.py -c $bpe_prefix.$lang2 < $train_file_prefix.$lang2 > $train_file_prefix.bpe.42k.$lang2

echo "clean the train files"
$CLEAN $train_file_prefix.bpe.42k $lang1 $lang2 $train_file_prefix.bpe.42k.clean 5 150 retained


echo "shuffle the train files"
python /home/ming/code/THUMT/thumt/scripts/shuffle_corpus.py --corpus $train_file_prefix.bpe.42k.clean.$lang1 $train_file_prefix.bpe.42k.clean.$lang2


echo "generating vocab..."
python /home/ming/code/THUMT/thumt/scripts/build_vocab.py $train_file_prefix.bpe.42k.clean.$lang1 vocab.42k.$lang1
python /home/ming/code/THUMT/thumt/scripts/build_vocab.py $train_file_prefix.bpe.42k.clean.$lang2 vocab.42k.$lang2

echo "processing  "$valid_file_prefix.$lang1 " , "$test_file_prefix.$lang1" ... "

python /home/ming/tools/subword-nmt/subword_nmt/apply_bpe.py -c $bpe_prefix.$lang1 < $valid_file_prefix.$lang1 > $valid_file_prefix.bpe.42k.$lang1
python /home/ming/tools/subword-nmt/subword_nmt/apply_bpe.py -c $bpe_prefix.$lang1 < $test_file_prefix.$lang1 > $test_file_prefix.bpe.42k.$lang1

