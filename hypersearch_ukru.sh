nclusters=$1
nepochs=500

python cluster.py -k $nclusters

CUDA_VISIBLE_DEVICES='' python train_refactored.py --model baseline --w2v ../data/uk-ru.w2v --batch_size 32 --num_epochs $nepochs
out=ncl${nclusters}-baseline-nepochs${nepochs}
mkdir -p $out; mv baseline.* $out/;cp kmeans.pickle $out/
python evaluate.py --w2v ../data/uk-ru.w2v $out &> $out/results.txt &

for lambdac in 0.03 0.1 0.3 1.0 ; do
  CUDA_VISIBLE_DEVICES='' python train_refactored.py --model toyota --w2v ../data/uk-ru.w2v --batch_size 32 --num_epochs $nepochs --lambdac $lambdac
  out=ncl${nclusters}-toyota-nepochs${nepochs}-lambdac${lambdac}
  mkdir -p $out; mv toyota.* $out/; cp kmeans.pickle $out/
  python evaluate.py --w2v ../data/uk-ru.w2v $out &> $out/results.txt &
  
done
