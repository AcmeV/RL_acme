python main.py \
--device=gpu \
--pre-training=1 \
--if-save=1 \
--model=PolicyGradient \
--env-type=gym \
--env-name=PendulumContinue \
--episodes=1000 \
--is-render=0 \
--lr=0.01

python main.py \
--device=gpu \
--pre-training=1 \
--if-save=1 \
--model=ActorCritic \
--env-type=gym \
--env-name=PendulumContinue \
--episodes=1000 \
--is-render=0 \
--lr=0.01
