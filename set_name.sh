#!/bin/bash

# Check if script was sourced and exit with an error if that is the case
if [ -n "$BASH_VERSION" ]; then
    (return 0 2>/dev/null) && sourced=1 || sourced=0
else
    echo "Please run this script using bash."
    return 1
fi

if [ "$sourced" -eq 1 ]; then
    echo "Please do not source this script."
    return 1
fi

# Sets the stage name (i.e. make this your own) and then deletes itself

read -p "New stage name (lower_snake_case): " stage_name_lsc
read -p "New stage name (UpperCamelCase): " stage_name_ucc
read -p "New stage name (nospaces): " stage_name_ns
read -p "New stage name (kebab-case): " stage_name_kc

if [[ -z "$stage_name_lsc" || -z "$stage_name_ucc" || -z "$stage_name_ns" || -z "$stage_name_kc" ]]; then
    echo "Name cannot be empty!"
    exit 1
fi

echo "Setting name to $stage_name_ucc/$stage_name_lsc/$stage_name_ns/$stage_name_kc"

sed -i "s/MyStage/$stage_name_ucc/" mystage/*.py
sed -i "s/my_stage/$stage_name_lsc/" mystage/*.py
sed -i "s/mystage/$stage_name_ns/" mystage/*.py pyproject.toml settings.template.yaml main.py
sed -i "s/my-stage/$stage_name_kc/" docker_build.sh docker_push.sh
mv mystage/ "$stage_name_ns/"
mv "$stage_name_ns/mystage.py" "$stage_name_ns/$stage_name_ns.py"

echo "# SAE $stage_name_kc" > README.md

echo "Done. Deleting myself..."

# Delete this script
rm -- "$0"