#!/usr/bin/env bash

cd $(dirname $(readlink -f $0))


unzip -u ../docs/source/tutorial/tutorial_example_scripts.zip
cd pyoomph_tutorial_scripts

function TestFolder()
{
(
 ALLGOOD=true
 echo "TESTING FOLDER $1" 
 cd $1
 for f in *.py; do
   od=${f%.py}
   if [[ "$f" != "bifurcation_fold_param_change.py" ]]; then
   echo "  testing: $f in $od"
   if python $f  &>${od}.log; then
    :
   else
     echo   "=========FAILED $f"
     ALLGOOD=false
   fi
   rm -rf "$od"
   fi
   
 done
 if $ALLGOOD; then
  :
 else
  echo "============ SOME TESTS IN $1 FAILED =============="
 fi
 echo
 echo 
)
}

for d in */; do
TestFolder $d
done

