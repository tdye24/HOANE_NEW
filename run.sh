for pretrain_wd in 5e-4 1e-3; do
  for finetune_wd in 1e-3; do
    for pretrain_dropout in 0.1 0.3 0.5; do
      for finetune_dropout in 0.3 0.5 0.7 0.9; do
        for pretrain_lr in 0.001; do
          for finetune_lr in 0.001; do
            for node_attr_attention_dropout in 0.0; do
              for encoder_layers in 2; do
                for decoder_layers in 2; do
                  for aug_e in 0.0; do
                    for aug_a in 0.1 0.2 0.3 0.4 0.5; do
                      python run.py \
                      --version v2 \
                      --pretrain-epochs 300 \
                      --finetune-epochs 500 \
                      --pretrain-lr $pretrain_lr \
                      --finetune-lr $finetune_lr \
                      --finetune-interval 30 \
                      --pretrain-dropout $pretrain_dropout \
                      --finetune-dropout $finetune_dropout \
                      --pretrain-wd $pretrain_wd \
                      --finetune-wd $finetune_wd \
                      --node-classification \
                      --num-hidden 512 \
                      --out-dim 512 \
                      --encoder-type gcn \
                      --encoder-layers $encoder_layers \
                      --decoder-type gcn \
                      --decoder-layers $decoder_layers \
                      --attr-loss-type bce \
                      --node-attr-attention-dropout $node_attr_attention_dropout \
                      --aug-e $aug_e \
                      --aug-a $aug_a \
                      --filename v3_result_auge0.txt
                    done
		              done
                done
              done
            done
          done
        done
      done
    done
  done
done
