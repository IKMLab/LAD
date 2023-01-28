exp_name=teacher_base
device_id=0
task=mnli

python run_fine_tune.py          \
--experiment $exp_name           \
--ptrain_ver bert-base-uncased   \
--task $task                     \
--dataset train                  \
--num_class 3                    \
--accum_step 1                   \
--batch_size 32                  \
--ckpt_step 3000                 \
--log_step 500                   \
--lr 3e-5                        \
--max_seq_len 128                \
--device_id $device_id           \
--seed 42                        \
--total_step 36816               \
--warmup_step 10000

python run_fine_tune_eval.py     \
--experiment $exp_name           \
--task $task                     \
--dataset dev_matched            \
--batch_size 128

python run_fine_tune_eval.py     \
--experiment $exp_name           \
--task $task                     \
--dataset dev_mismatched         \
--batch_size 128