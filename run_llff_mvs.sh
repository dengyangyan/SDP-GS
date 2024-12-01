scene_path="/home/dengyangyan/code/data/nerf_llff_data"
op="output/llff_fsgs"
declare -a arr=("flower" "fern" "horns" "leaves" "orchids" "room" "trex" "fortress")

for i in ${arr[@]}; do
    dataset="$scene_path/$i"
    workspace="$op/$i"

    python train.py -s $dataset --model_path $workspace -r 8 --nviews 3 --lambda_dssim 0.2 \
                --eval \
                --random_background \
                --iterations 10000 --position_lr_max_steps 10000 \
                --densify_until_iter 10000 \
                --densify_grad_threshold 0.0005 \
                --sample_pseudo_interval 1 --end_sample_pseudo 10000 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --scaling_lr 0.005 \
                --save_iterations 100 500 1000 3000 6000 8000 10000\
                --checkpoint_iterations 10000 \
                --sample_pseudo_interval 1 --start_sample_pseudo 2000 --end_sample_pseudo 9500 \
                --depth_pseudo_weight 0.5 \
                --prune_threshold 0.005 \
                --include_feature


done

# # set a larger "--error_tolerance" may get more smooth results in visualization
          

for i in ${arr[@]}; do
    dataset="$scene_path/$i"
    workspace="$op/$i"
    python render.py -s $dataset --model_path $workspace -r 8 --iteration 10000 --render_depth
    python metrics.py --model_path $workspace 
done

python metric_.py --model_path $op

        

