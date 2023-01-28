device_id=0
task=mrpc

python run_lad_distil.py           \
--teacher_exp teacher_base         \
--tckpt 345                        \
--experiment LAD_6_layer           \
--task $task                       \
--batch_size 32                    \
--ckpt_step 50                     \
--log_step 10                      \
--d_ff 3072                        \
--d_model 768                      \
--lr 7e-4                          \
--gate_lr 5e-6                     \
--num_hidden_layers 6              \
--epoch 20                         \
--warmup_rate 0.1                  \
--device_id $device_id             \
--tdevice_id $device_id            \
--gate_device_id $device_id        \
--seed 42                          \
--softmax_temp 10                  \
--soft_weight 0.5                  \
--hard_weight 0.5                  \
--hidden_mse_weight 500

python run_fine_tune_eval.py                   \
--experiment LAD_6_layer                       \
--task $task                                   \
--dataset dev                                  \
--batch_size 512                               \
--device_id $device_id                         \
--do_predict