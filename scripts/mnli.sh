device_id=0
task=mnli

python run_lad_distil_epoch.py     \
--teacher_exp teacher_base         \
--tckpt 36816                      \
--experiment LAD_6_layer           \
--task $task                       \
--batch_size 32                    \
--ckpt_step 2000                   \
--log_step 200                     \
--d_ff 3072                        \
--d_model 768                      \
--lr 1e-4                          \
--gate_lr 5e-7                     \
--num_hidden_layers 6              \
--epoch 4                          \
--warmup_rate 0.1                  \
--device_id $device_id             \
--tdevice_id $device_id            \
--gate_device_id $device_id        \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                  \
--hidden_mse_weight 500

python run_fine_tune_eval.py       \
--experiment LAD_6_layer           \
--task $task                       \
--dataset dev_matched              \
--batch_size 512                   \
--device_id $device_id             \
--do_predict

python run_fine_tune_eval.py       \
--experiment LAD_6_layer           \
--task mnli                        \
--dataset dev_mismatched           \
--batch_size 512                   \
--do_predict