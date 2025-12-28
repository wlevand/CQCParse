#!/bin/bash
################### Gaussian Job Batch Script Example ###################
# Section for defining queue-system variables:
#-------------------------------------
# SLURM-section
#SBATCH --account=nn14654k
#SBATCH --job-name=co/STO-3G
#SBATCH --nodes=1 --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=5GB
######################################
# Section for defining job variables and settings:
#-------------------------------------

#input=RC003_b3lyp_upc1 # Name of input without extention
#extention=com # We use the same naming scheme as the software default

module load Gaussian/g16_B.01

# This one is important; setting the heap-size for the job to 20GB:
export GAUSS_LFLAGS2="--LindaOptions -s 20000000"
#Creating the gaussian temp dir:
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# Creating aliases:
submitdir=$SLURM_SUBMIT_DIR
tempdir=$GAUSS_SCRDIR

# split large temporary files into smaller parts:
lfs setstripe –c 8 $tempdir

# Moving files to scratch:
cd $submitdir
cp $1.com $tempdir
cp $1.chk $tempdir
cd $tempdir

# Running program, pay attention to command name:

time g16.ib $1.com > g16_$1.out

# Cleaning up and moving files back to home/submitdir:

cp g16_$1.out $submitdir
cp $1.chk $submitdir

exit 0


