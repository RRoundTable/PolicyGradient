
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.initializers import RandomNormal

# Agent
class PolicyGradientAgent(object):
    def __init__(self, hparams, sess):
        # initialization
        self._s=sess

        # build the graph : policy
        self._input=tf.placeholder(tf.float32, shape=[None, hparams['input_size']]) # observations

        hidden1 = Dense(
            units=hparams['hidden_size'],
            activation=tf.nn.relu,
            kernel_initializer=RandomNormal)(self._input)

        logits = Dense(
            units=hparams['num_actions'],
            activation=None)(hidden1)

        # op to sample an action
        self._sample=tf.reshape(tf.multinomial(logits,1),[]) # draw one sample from multinomial distribution
        self._sample_=tf.multinomial(logits,1)
        self._sample_=tf.squeeze(self._sample_)
        # get log probabilities
        self.log_prob=tf.log(tf.nn.softmax(logits))

        # training prat of graph
        self._acts=tf.placeholder(tf.int32)
        self._advantages=tf.placeholder(tf.float32)

        # get log probs of actions from episodes
        self.indices=tf.range(0, tf.shape(self.log_prob)[0])*tf.shape(self.log_prob)[1]+self._acts
        self.act_prob=tf.gather(tf.reshape(self.log_prob, [-1]), self.indices) # tf.gather : arrangement

        # surrogate loss
        self.loss=-tf.reduce_sum(tf.multiply(self.act_prob, self._advantages))

        # update
        optimizer=tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train=optimizer.minimize(self.loss)

    def act(self, observation):
        """ return action from distribution """
        print("-----------sample----------")
        print(self._s.run(self._sample, feed_dict={self._input: [observation]}))
        return self._s.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self,obs,acts,advantages):
        batch_feed={self._input:obs,
                    self._acts:acts,
                    self._advantages:advantages}
        print("acts : {}".format(acts))
        # train
        self._s.run(self._train,feed_dict=batch_feed)
        loss, act_prob=self._s.run([self.loss, self.act_prob],feed_dict=batch_feed)
        indices = self._s.run([self.indices], feed_dict={self._input:obs,
                                                         self._acts : acts})

        samples, log_prob=self._s.run([self._sample_, self.log_prob], feed_dict={self._input :obs}) # error
        print("log prob : {}".format(log_prob))
        print("log prob : {}".format(log_prob.shape))
        print(" loss : {}".format(loss))
        print(" act_prob : {}".format(act_prob))
        print(" act_prob : {}".format(len(act_prob)))
        print(" indices : {}".format(indices))
        print(" indices : {}".format(len(indices[0])))
        print(" samples : {}".format(samples))
        #print(" samples : {}".format(len(samples[0])))


def policy_rollout(env, agent):

    """
    run one episodes
    :param env: enviroment
    :param agent: agent
    :return:  obs, acts, rews
    """
    # env reset
    observation,reward, done= env.reset(), 0, False
    obs, acts, rews =[],[],[]

    while not done: # until one episode ends
        env.render()
        obs.append(observation)

        action=agent.act(observation) # self._sample
        observation, reward, done, _ =env.step(action)

        acts.append(action)
        rews.append(reward)
    return obs, acts, rews



def progress_rewards(rews):

    """
     rewards -> Advantages for one episode
     짧게 끝날 수록 좋은 것이다
    """
    return [len(rews)]*len(rews)


def main():

    env = gym.make('CartPole-v0')


    # monitor_dir='/tmp/cartpole_exp1'
    # # env.monitoring.start(monitor_dir)
    #
    # env=gym.wrappers.Monitor(env,monitor_dir, force=True)
    # hyper parameters
    hparams={
        'input_size' : env.observation_space.shape[0],
        'hidden_size': 36,
        'num_actions' : env.action_space.n,
        'learning_rate' : 0.1
    }

    # enviroment params
    eparams={
        'num_batches' :40,
        'ep_per_batch' : 10
    }

    with tf.Session() as sess:
        agent=PolicyGradientAgent(hparams,sess)
        sess.run(tf.initialize_all_variables())

        for batch in range(eparams['num_batches']):
            print("=======\n BATCH {}\n======".format(batch))

            b_obs, b_acts, b_rews=[],[],[]

            for _ in range(eparams['ep_per_batch']):
                # episode
                obs, acts, rews=policy_rollout(env,agent)

                print(" Episode steps : {}".format(len(obs)))

                b_obs.extend(obs)
                b_acts.extend(acts)

                advantages=progress_rewards(rews)
                b_rews.extend(advantages)



            # update Policy
            # normalize rewards : don't devide by 0
            b_rews=(b_rews-np.mean(b_rews))/(np.std(b_rews)+1e-10)
            agent.train_step(b_obs, b_acts,b_rews)





        env.monitor.close()
main()

if '__name__'=="__main__":
    main()

