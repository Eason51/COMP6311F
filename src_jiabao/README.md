
# README

How to run the codes? (By yicheng)

python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --epochs=20 --local_ep=1 --frac=1 --num_users=5 --scaffold=1 --weighted=0

--iid=0，就是non-iid分布，--iid=1，就是iid分布，
--scaffold=1就是用scaffold算法，
--weighted=1就是加权平均参数，weighted如果是1，scaffold也必须是1