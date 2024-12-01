base="/home/dengyangyan/code/experiment/FSGS-main_total/segment-aware-GS/our_gaussian-splatting/output/dtu_ours_stereo_stereo_novelview_01"
mask_path="/home/dengyangyan/code/data/idrmasks"

for scan_id in scan34 scan41 scan45 scan103  scan38  scan21 scan40  scan55  scan63  scan31  scan8  scan114
# for scan_id in scan114
do  
    if [ -d $base/$scan_id ]; then
        # rm -r $base/$scan_id/mask
        mkdir $base/$scan_id/mask
        id=0
        if [ -d ${mask_path}/$scan_id/mask ]; then
            for file in ${mask_path}/scan8/*
            do  
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp ${file//scan8/$scan_id'/mask'} $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done

            else

            for file in ${mask_path}/$scan_id/*
            do
                # echo $file
                file_name=$(printf "%05d" $id).png;
                cp $file $base/$scan_id/mask/$file_name
                ((id = id + 1))
            done
        fi
    fi
    
done