#!/bin/bash

seed=$1
ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda no  --seed"
PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda no  --seed"
MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 1 --log_every 100 --samples_per_task 200 --data_file mnist_manypermutations.pt --cuda no  --seed"

echo "Beginning Online Learning" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model online --lr 0.0003
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model online --lr 0.003
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model online --lr 0.003

echo "Beginning Independent Model Per Task" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model independent --lr 0.01
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model independent --lr 0.01
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model independent --lr 0.01

echo "Beginning EWC" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model ewc --lr 0.001 --n_memories 10 --memory_strength 100.0
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model ewc --lr 0.01 --n_memories 10 --memory_strength 10.0
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model ewc --lr 0.003 --n_memories 10 --memory_strength 1.0

echo "Beginning GEM With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model gem --lr 0.01 --n_memories 256 --memory_strength 1.0
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model gem --lr 0.01 --n_memories 256 --memory_strength 1.0
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model gem --lr 0.01 --n_memories 51 --memory_strength 0.0

echo "Beginning GEM With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model gem --lr 0.01 --n_memories 25 --memory_strength 0.0
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model gem --lr 0.01 --n_memories 25 --memory_strength 1.0
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model gem --lr 0.003 --n_memories 5 --memory_strength 0.1

echo "Beginning GEM With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model gem --lr 0.01 --n_memories 10 --memory_strength 0.0
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model gem --lr 0.01 --n_memories 10 --memory_strength 0.0

echo "Beginning ER (Algorithm 4) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg4 --lr 0.1 --memories 5120 --replay_batch_size 25
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg4 --lr 0.1 --memories 5120 --replay_batch_size 25
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model eralg4 --lr 0.1 --memories 5120 --replay_batch_size 25

echo "Beginning ER (Algorithm 4) With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg4 --lr 0.1 --memories 500 --replay_batch_size 5
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg4 --lr 0.1 --memories 500 --replay_batch_size 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model eralg4 --lr 0.1 --memories 500 --replay_batch_size 25

echo "Beginning ER (Algorithm 4) With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg4 --lr 0.1 --memories 200 --replay_batch_size 10
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg4 --lr 0.1 --memories 200 --replay_batch_size 10  

echo "Beginning ER (Algorithm 5) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg5 --lr 0.03 --memories 5120 --replay_batch_size 100
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg5 --lr 0.01 --memories 5120 --replay_batch_size 25
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model eralg5 --lr 0.003 --memories 5120 --replay_batch_size 10

echo "Beginning ER (Algorithm 5) With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg5 --lr 0.01 --memories 500 --replay_batch_size 100
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg5 --lr 0.01 --memories 500 --replay_batch_size 25
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model eralg5 --lr 0.01 --memories 500 --replay_batch_size 5

echo "Beginning ER (Algorithm 5) With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model eralg5 --lr 0.01 --memories 200 --replay_batch_size 50
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model eralg5 --lr 0.01 --memories 200 --replay_batch_size 10

echo "Beginning MER (Algorithm 1) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg1 --lr 0.1 --beta 0.01 --gamma 1.0 --memories 5120 --replay_batch_size 5 --batches_per_example 10

echo "Beginning MER (Algorithm 1) With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg1 --lr 0.1 --beta 0.01 --gamma 1.0 --memories 500 --replay_batch_size 10 --batches_per_example 10
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 500 --replay_batch_size 25 --batches_per_example 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 500 --replay_batch_size 5 --batches_per_example 10

echo "Beginning MER (Algorithm 1) With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg1 --lr 0.1 --beta 0.01 --gamma 1.0 --memories 200 --replay_batch_size 10 --batches_per_example 5
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 200 --replay_batch_size 10 --batches_per_example 10

echo "Beginning MER (Algorithm 6) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg6 --lr 0.03 --gamma 0.1 --memories 5120 --replay_batch_size 100
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg6 --lr 0.03 --gamma 0.1 --memories 5120 --replay_batch_size 50
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg6 --lr 0.03 --gamma 0.1 --memories 5120 --replay_batch_size 25

echo "Beginning MER (Algorithm 6) With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg6 --lr 0.1 --gamma 0.03 --memories 500 --replay_batch_size 10
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg6 --lr 0.03 --gamma 0.3 --memories 500 --replay_batch_size 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg6 --lr 0.1 --gamma 0.03 --memories 500 --replay_batch_size 5

echo "Beginning MER (Algorithm 6) With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg6 --lr 0.1 --gamma 0.03 --memories 200 --replay_batch_size 25
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg6 --lr 0.1 --gamma 0.03 --memories 200 --replay_batch_size 5


echo "Beginning MER (Algorithm 7) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 5120 --replay_batch_size 100 --s 5
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg7 --lr 0.01 --gamma 0.1 --memories 5120 --replay_batch_size 100 --s 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 5120 --replay_batch_size 100 --s 10

echo "Beginning MER (Algorithm 7) With 500 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 500 --replay_batch_size 50 --s 5
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg7 --lr 0.01 --gamma 0.1 --memories 500 --replay_batch_size 25 --s 10
echo "MNIST Many Permutations:"
python3 main.py $MANY $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 500 --replay_batch_size 5 --s 10

echo "Beginning MER (Algorithm 7) With 200 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model meralg7 --lr 0.03 --gamma 0.03 --memories 200 --replay_batch_size 50 --s 5
echo "MNIST Permutations:"
python3 main.py $PERM $seed --model meralg7 --lr 0.03 --gamma 0.1 --memories 200 --replay_batch_size 5 --s 2
