#!/bin/bash

seed=0
# Test on a small number of samples per task
ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 10 --data_file mnist_rotations.pt    --cuda no  --seed"

echo "Begin Unit Test with seed =" $seed " on MNIST Rotations"
echo "Testing Online"
python3 main.py $ROT $seed --model online --lr 0.0003
if [ $? -eq 0 ]
then
  echo "Online finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi

echo "Testing Independent"
python3 main.py $ROT $seed --model independent --lr 0.01
if [ $? -eq 0 ]
then
  echo "Independent finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi

echo "Testing EWC"
python3 main.py $ROT $seed --model ewc --lr 0.001 --n_memories 10 --memory_strength 100.0
if [ $? -eq 0 ]
then
  echo "EWC finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi

echo "Testing GEM"
python3 main.py $ROT $seed --model gem --lr 0.01 --n_memories 256 --memory_strength 1.0
if [ $? -eq 0 ]
then
  echo "GEM finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Testing eralg4"
python3 main.py $ROT $seed --model eralg4 --lr 0.1 --memories 5120 --replay_batch_size 25
if [ $? -eq 0 ]
then
  echo "eralg4 finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Testing eralg5"
python3 main.py $ROT $seed --model eralg5 --lr 0.03 --memories 5120 --replay_batch_size 100
if [ $? -eq 0 ]
then
  echo "eralg5 finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Testing meralg1"
python3 main.py $ROT $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10
if [ $? -eq 0 ]
then
  echo "meralg1 finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Testing meralg6"
python3 main.py $ROT $seed --model meralg6 --lr 0.03 --gamma 0.1 --memories 5120 --replay_batch_size 100
if [ $? -eq 0 ]
then
  echo "meralg6 finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Testing meralg7"
python3 main.py $ROT $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 500 --replay_batch_size 50 --s 5
if [ $? -eq 0 ]
then
  echo "meralg7 finished successfully"
else
  echo "Exited with error." >&2  # Redirect stdout from echo command to stderr.
fi


echo "Finished"
