PROJECT_ROOT="/home/bilkit/Workspace/mj_envs_vision/results/supercloud"
SUPER_ROOT="/home/gridsan/bgithinji/mj_envs_vision/results"
SUPER_HOST="bgithinji@txe1-login.mit.edu"

echo "copying new results from"
echo "$SUPER_HOST:$SUPER_ROOT"
echo "to"
echo "$PROJECT_ROOT"

# copy new results
echo "" > "$PROJECT_ROOT/copy.log"
mkdir -p "$PROJECT_ROOT"
for item in $(ssh bgithinji@txe1-login.mit.edu "ls $SUPER_ROOT");do #$(rsync -r $SUPER_ROOT/results);do #/tmp/sc/*;do
  if [[ ! -e "$PROJECT_ROOT/$item" ]];then
    echo "+ $item"
    scp -r "$SUPER_HOST:$SUPER_ROOT/$item" $PROJECT_ROOT >> $PROJECT_ROOT/copy.log
  fi
done
