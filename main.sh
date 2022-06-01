python main.py \
--device=gpus \
--gpus=0,1,2,3 \
--pre-training=1 \
--if-save=1 \
--model=QLearning \
--env-type=tkinter \
--env-name=Maze \
--episodes=150 \
--lr=0.1
