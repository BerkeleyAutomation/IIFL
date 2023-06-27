# BC C
for i in {1000,2000,3000}
do
    CUDA_VISIBLE_DEVICES=1 python -m train @scripts/args_common.txt \
    --env_name Ant --logdir_suffix C --seed $i --allocation CUR \
    --order C --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 1 --use_player 0 --updates_per_step 0
done

# BC UC
for i in {1000,2000,3000}
do
    CUDA_VISIBLE_DEVICES=1 python -m train @scripts/args_common.txt \
    --env_name Ant --logdir_suffix UC --seed $i --allocation CUR \
    --order UC --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 1 --use_player 0 --updates_per_step 2 --uncertainty_thresh 0.0304
done

# IBC C
for i in {1000,2000,3000}
do
    CUDA_VISIBLE_DEVICES=1 python -m train @scripts/args_common.txt \
    --env_name Ant --logdir_suffix C --seed $i --allocation CUR \
    --order C --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 1 --use_player 0 --updates_per_step 0 --agent IBC
done

# IBC UC
for i in {1000,2000,3000}
do
    CUDA_VISIBLE_DEVICES=1 python -m train @scripts/args_common.txt \
    --env_name Ant --logdir_suffix UC --seed $i --allocation CUR --order UC \
    --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 1 --use_player 0 --updates_per_step 2000 --update_every 1000 \
    --agent IBC --uncertainty_thresh=0.9062
done

# IBC R
for i in {1000,2000,3000}
do
    CUDA_VISIBLE_DEVICES=1 python -m train @scripts/args_common.txt \
    --env_name Ant --logdir_suffix R --seed $i --allocation random \
    --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 1 --use_player 0 --updates_per_step 2000 --update_every 1000 \
    --agent IBC --action_budget 20000
done