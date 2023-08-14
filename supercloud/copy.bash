PROJ_ROOT="/home/bilkit/Workspace/mj_envs_vision"
SUPER_ROOT="bgithinji@txe1-login.mit.edu:/home/gridsan/bgithinji"
IGNORE_FILE="$PROJ_ROOT/.dockerignore" # WARN: make sure all dirs end with '/'

echo "this script will copy items to $SUPER_ROOT/mj_envs_vision and ignore items listed in $IGNORE_FILE" 
echo "overwritting $SUPER_ROOT/mj_envs_vision"
ssh bgithinji@txe1-login.mit.edu  "rm -rf /home/gridsan/bgithinji/mj_envs_vision"
ssh bgithinji@txe1-login.mit.edu  "mkdir  /home/gridsan/bgithinji/mj_envs_vision"
echo ""

echo "" > copy.log
for item in $PROJ_ROOT/*;do
  if [[ -d $item ]];then
    if [[ $(cat $IGNORE_FILE | grep -e "^$(basename $item)\/$") ]];then
      echo "- $item" 
    else
      echo "+ $item"
      scp -r $item  $SUPER_ROOT/mj_envs_vision >> copy.log
    fi
  elif [[ -f $item ]];then
    if [[ $(cat $IGNORE_FILE | grep -e "^$(basename $item)$") || ${item##*.} == "json" ]];then
      echo "- $item" 
    else
      echo "+ $item"
      scp $item  $SUPER_ROOT/mj_envs_vision >> copy.log
    fi
  fi
 done

# Exceptions to ignore file
echo ""
echo "copying dependencies"
echo ""
if [[ $(cat $IGNORE_FILE | grep -e "^dependencies/") ]];then
  echo "+ $PROJ_ROOT/dependencies"
  scp -r $PROJ_ROOT/dependencies  $SUPER_ROOT/mj_envs_vision >> copy.log
fi

echo "overwritting $SUPER_ROOT/.mujoco"
ssh bgithinji@txe1-login.mit.edu  "rm -rf /home/gridsan/bgithinji/.mujoco"
ssh bgithinji@txe1-login.mit.edu  "mkdir  /home/gridsan/bgithinji/.mujoco"
echo "+ ~/.mujoco/mujoco210"
scp -r ~/.mujoco/mujoco210  $SUPER_ROOT/.mujoco >> copy.log

#if [[ ${grep -e "*" <<< $item} ]];then
