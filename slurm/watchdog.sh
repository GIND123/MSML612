#!/bin/bash
# Safety net: re-runs finalize periodically, independent of any grid.
#
# finalize.sh is chained to each grid, but that only fires when a grid ends
# cleanly. If a grid is cancelled, times out, or the chained job itself fails,
# results would sit on scratch un-published. This resubmits itself every few
# hours so collection and the Hugging Face push keep happening regardless of
# what the laptop, the network, or the login session are doing.
#
# Start once with:   sbatch watchdog.sh
# Stop with:         scancel -n watchdog
#
#SBATCH -J watchdog
#SBATCH -t 00:45:00
#SBATCH -c 2
#SBATCH --mem=8g
#SBATCH -p standard
#SBATCH -o logs/watchdog-%j.out

SCR=/scratch/zt1/project/msml612/user/$USER
cd $SCR/pcmd

RUNS_NOW=$(ls $SCR/pcmd/runs 2>/dev/null | wc -l)
LAST=$(cat $SCR/pcmd/.watchdog_last 2>/dev/null || echo 0)
echo "$(date): runs=$RUNS_NOW last_published=$LAST"

# Only spend effort when something actually changed.
if [ "$RUNS_NOW" -ne "$LAST" ]; then
  echo "new results detected - running finalize"
  bash $SCR/pcmd/finalize.sh
  echo "$RUNS_NOW" > $SCR/pcmd/.watchdog_last
else
  echo "nothing new since last publish"
fi

# Re-arm unless someone asked it to stop.
if [ ! -f $SCR/pcmd/.watchdog_stop ]; then
  sbatch --begin=now+3hours $SCR/pcmd/watchdog.sh
  echo "re-armed for +3h"
else
  echo "stop file present - not re-arming"
fi
