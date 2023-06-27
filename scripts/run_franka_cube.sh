# explicit BC
for i in {1000,2000,3000}
do
    python -m train @scripts/args_franka_cube.txt \
    --logdir_suffix C --seed $i --allocation CUR \
    --order C --num_task_transitions 10000 --policy_pretraining_steps 1000 --num_envs 100 \
    --num_humans 10 --num_players 2 --updates_per_step 0
done

# explicit IFL
for i in {1000,2000,3000}
do
    python -m train @scripts/args_franka_cube.txt \
    --logdir_suffix UC --seed $i --allocation CUR \
    --order UC --num_task_transitions 10000 --policy_pretraining_steps 1000 --num_envs 100 \
    --num_humans 10 --num_players 2 --updates_per_step 0 --uncertainty_thresh=0.0 --no_free_humans
done

# implicit BC
for i in {1000,2000,3000}
do
    python -m train @scripts/args_franka_cube.txt \
    --logdir_suffix C --seed $i --allocation CUR \
    --order C --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 2 --agent IBC --updates_per_step 0 \
    --update_every 1000
done

# random implicit IFL
for i in {1000,2000,3000}
do
    python -m train @scripts/args_franka_cube.txt \
    --logdir_suffix R --seed $i --allocation random \
    --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 2 --agent IBC --updates_per_step 0 \
    --update_every 1000 --no_free_humans
done

# implicit IFL
for i in {1000,2000,3000}
do
    python -m train @scripts/args_franka_cube.txt \
    --logdir_suffix UC --seed $i --allocation CUR \
    --order UC --num_task_transitions 10000 --policy_pretraining_steps 5000 --num_envs 100 \
    --num_humans 10 --num_players 2 --agent IBC --updates_per_step 0 \
    --update_every 1000 --uncertainty_thresh=0.0 --no_free_humans
done
