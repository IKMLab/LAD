device_id=0
task=qqp

python run_lad_distil_epoch.py     \
--teacher_exp teacher_base         \
--tckpt 34110                      \
--experiment LAD_6_layer           \
--task $task                       \
--batch_size 32                    \
--ckpt_step 3000                   \
--log_step 100                     \
--d_ff 3072                        \
--d_model 768                      \
--lr 5e-4                          \
--gate_lr 5e-7                     \
--num_hidden_layers 6              \
--epoch 5                          \
--warmup_rate 0.1                  \
--device_id $device_id             \
--tdevice_id $device_id            \
--gate_device_id $device_id        \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.7                  \
--hard_weight 0.3                  \
--hidden_mse_weight 1000

python run_fine_tune_eval.py       \
--experiment LAD_6_layer           \
--task $task                       \
--dataset dev                      \
--batch_size 512                   \
--device_id $device_id             \
--do_predict