for pretrain_wd in 0 5e-5 1e-4 5e-4 1e-3
do
  for finetune_wd in 0 5e-5 1e-4 5e-4 1e-3
  do
    for pretrain_dropout in 0 0.1 0.3 0.5 0.7 0.9
    do
      for finetune_dropout in 0 0.1 0.3 0.5 0.7 0.9
      do
        for pretrain_lr in 0.001
        do
          for finetune_lr in 0.001 0.005 0.01
          do
            python run.py \
            --pretrain-epochs 100 \
            --finetune-epochs 500 \
            --pretrain-lr $pretrain_lr \
            --finetune-lr $finetune_lr \
            --finetune-interval 10 \
            --pretrain-dropout $pretrain_dropout \
            --finetune-dropout $finetune_dropout \
            --pretrain-wd $pretrain_wd \
            --finetune-wd $finetune_wd \
            --node-classification
          done
        done
      done
    done
  done
done