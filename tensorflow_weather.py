
# %tensorflow_version 2.x
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8,0.2])
trans_distribution = tfd.Categorical(probs=[[0.9,0.1],[0.2,0.8]])
observ_distribution = tfd.Normal(loc=[0., 15.], scale=[5.,10.])

model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution, transition_distribution=trans_distribution, observation_distribution=observ_distribution,num_steps=14)

mean=model.mean()

with tf.compat.v1.Session() as se:
  print(mean.numpy())
