exp_name=teacher_base
device_id=0
task=mrpc

python run_fine_tune.py          \
--experiment $exp_name           \
--ptrain_ver bert-base-uncased   \
--task $task                     \
--dataset train                  \
--num_class 2                    \
--accum_step 1                   \
--batch_size 32                  \
--ckpt_step 115                  \
--log_step 10                    \
--lr 7e-5                        \
--max_seq_len 128                \
--device_id $device_id           \
--seed 42                        \
--total_step 345                 \
--warmup_step 1

python run_fine_tune_eval.py     \
--experiment $exp_name           \
--task $task                     \
--dataset dev                    \
--batch_size 128