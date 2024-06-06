#!/usr/bin/bash

#SBATCH --job-name=MOLPPS
#SBATCH --time=2:00:00 
#SBATCH --mem-per-cpu=35000mb
#SBATCH --ntasks=5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bert.verbruggen@vub.be
#SBATCH --output MOL_PPr.log
 
cd $SLURM_SUNMIT_DIR
echo Start Job
date
 
ml matplotlib/3.4.3-foss-2021b
 
# Create a unique directory for this run
OUTPUT_DIR= #Provide path to output directory $path$#
mkdir -p "$OUTPUT_DIR"

# Path to the CSV file
CSV_FILE= #Provide source path of hyperparameter csv file $path.csv$#
MODEL_FILE= #Provide path to model file $path.py$#
# Skip the header and read the file line by line
tail -n +3 "$CSV_FILE" | while IFS=, read -r Id Epochs Runs Discount learning_rate Epsilon reward1 reward2 reward3 epsilon_strategy lr_strategy epsilon_rate lr_rate epsilon_min lr_min
do
    #echo "Running with Id=$Id, Epochs=$Epochs, Runs=$Runs, Discount=$Discount, learning_rate=$learning_rate, Epsilon=$Epsilon, reward1=$reward1, reward2=$reward2, reward3=$reward3, epsilon_strategy=$epsilon_strategy, lr_strategy=$lr_strategy, epsilon_rate=$epsilon_rate, lr_rate=$lr_rate, epsilon_min=$epsilon_min, lr_min=$lr_min, output_dir=$OUTPUT_DIR"
    srun --ntasks=1 --exact python "$MODEL_FILE" "$Id" "$Epochs" "$Runs" "$Discount" "$learning_rate" "$Epsilon" "$reward1" "$reward2" "$reward3" "$epsilon_strategy" "$lr_strategy" "$epsilon_rate" "$lr_rate" "$epsilon_min" "$lr_min" "$OUTPUT_DIR" &
done

wait
 
echo "All processes completed!"