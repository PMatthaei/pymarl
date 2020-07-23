#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_pymarl_${HASH}

title="Select hardware optimization"
prompt="Pick:"
options=("Run on all GPUs" "Run on all CPUs")

echo "$title"
PS3="$prompt "
select opt in "${options[@]}" "Quit"; do

  case "$REPLY" in

  1)
    echo "You picked $opt which is option $REPLY"
    break
    ;;
  2)
    echo "You picked $opt which is option $REPLY"
    break
    ;;

  *)
    echo "Invalid option. Try another one."
    continue
    ;;

  esac

done

case "$REPLY" in

1)
  echo "Launching container named '${name}' on: all GPUs"
  sudo docker run \
  --gpus all \
  --name $name \
  --user $(id -u):$(id -g) \
  -v $(pwd):/pymarl \
  -t pymarl:1.0 \
  ${@:2}
  break
  ;;
2)
  echo "Launching container named '${name}' on: all CPUs"
  sudo docker run \
  --name $name \
  --user $(id -u):$(id -g) \
  -v $(pwd):/pymarl \
  -t pymarl:1.0 \
  ${@:2}
  ;;

*)
  echo "Invalid option. Try another one."
  continue
  ;;

esac