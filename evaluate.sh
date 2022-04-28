SCRIPTS=/home/ming/tools/mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/detokenizer.perl
BLEU=$SCRIPTS/generic/multi-bleu.perl
METEOR=/home/ming/tools/meteor-1.5/meteor-1.5.jar

lang1=en
lang2=de
predict=output.$lang2
refer=test.norm.clean.32k.$lang2

echo "DEBPE..."

sed -r 's/(@@ )|(@@ ?$)//g' < $predict > output.clean.$lang2
rm $predict

echo "DeTokenize..."
perl $DETOKENIZER -l $lang2 <output.clean.$lang2> output.clean.detok.$lang2

echo "Calculate BLEU Score..."
sacrebleu $refer -i output.clean.detok.$lang2 -m bleu -b -w 4
perl $BLEU -lc $refer < output.clean.detok.$lang2>

echo "Calculate METEOR Score..."
java -Xmx2G -jar $output.clean.detok.$lang2 $refer -norm -l $lang2