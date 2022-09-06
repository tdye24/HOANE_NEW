for pretrain_wd in 0; do
  for finetune_wd in 1e-3; do
    for pretrain_dropout in 0.1 0.3; do
      for finetune_dropout in 0.7 0.9; do
        for pretrain_lr in 0.001; do
          for finetune_lr in 0.001; do
            for node_attr_attention_dropout in 0.0; do
              for encoder_layers in 1 2 3; do
                for decoder_layers in 1 2 3; do
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
                  --node-classification \
                  --num-hidden 512 \
                  --out-dim 512 \
                  --encoder-type gcn \
                  --encoder-layers $encoder_layers \
                  --decoder-type gcn \
                  --decoder-layers $decoder_layers \
                  --attr-loss-type bce \
                  --node-attr-attention \
                  --node-attr-attention-dropout $node_attr_attention_dropout \
                  --filename gcn_num_layers.txt
                done
              done
            done
          done
        done
      done
    done
  done
done