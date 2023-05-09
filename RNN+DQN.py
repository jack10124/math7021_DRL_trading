
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import tensorflow.compat.v1 as tf
import os
import time




class Agent:
    #输出参数
    OUTPUT_SIZE = 3
    #超参数
    LEARNING_RATE = 0.003
    BATCH_SIZE = 32
    LAYER_SIZE = 256 #这里表示的是lstm网络子网的内层函数的神经元个数
    EPSILON = 0.5
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()#collection 模块提供的记忆单元（类似list结构)此时为初始化
    MEMORY_SIZE = 300 #最大记忆序列数量

    def __init__(self, state_size, window_size, trend, skip,mag):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.magicn=mag
        #绘图装饰器
        tf.reset_default_graph()
        self.INITIAL_FEATURES = np.zeros((self.magicn, self.state_size)) # 4 ：= state, action, reward, new_state  self.state_size=window长度
        self.X = tf.placeholder(tf.float32, (None, None, self.state_size))
        self.Y = tf.placeholder(tf.float32, (None, self.OUTPUT_SIZE))
        #变化核心
        cell = tf.nn.rnn_cell.LSTMCell(self.LAYER_SIZE, state_is_tuple = False)
        self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * self.LAYER_SIZE)) #这里遵循lstm基本逻辑使用了两个状态（细胞状态，隐藏状态）
        self.rnn ,self.last_state = tf.nn.dynamic_rnn(inputs=self.X ,cell=cell,
                                                     dtype=tf.float32,
                                                     initial_state=self.hidden_layer)

        #这里输出的 self.rnn 及有256个
        #输出逻辑层
        self.logits = tf.layers.dense(self.rnn[: ,-1], self.OUTPUT_SIZE)
        #sq err
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        #adam 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
        #模型初始化
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def _memorize(self, state, action, reward, new_state, dead, rnn_state):
        self.MEMORIES.append((state, action, reward, new_state, dead, rnn_state))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()#当记忆池记忆存满就会最懂删掉第一组记忆数量

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])#new state
        init_values = np.array([a[-1] for a in replay])#rnn state
        Q = self.sess.run(self.logits, feed_dict={self.X :states, self.hidden_layer :init_values})
        Q_new = self.sess.run(self.logits, feed_dict={self.X :new_states, self.hidden_layer :init_values})
        replay_size = len(replay)
        X = np.empty((replay_size, self.magicn, self.state_size))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
        INIT_VAL = np.empty((replay_size, 2 * self.LAYER_SIZE))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, dead_r, rnn_memory = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not dead_r:
                target[action_r] += self.GAMMA * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
            INIT_VAL[i] = rnn_memory
        return X, Y, INIT_VAL

    def get_state(self, t):
        #该函数将根据输入时间t 回访t时刻前windowsize天数的数据
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])#储存的是交易数据
        return np.array(res)

    def test(self, initial_money,test):
        print("//////////////////////////////////////////////////////////////////////////////////////////////test part :")
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        cash=[]
        holder=[]
        hold_num=0
        cash.append(initial_money)
        Journal = 'Test Trading Log：\n'
        state = self.get_state(0)
        init_value = np.zeros((1, 2 * self.LAYER_SIZE))
        for k in range(self.INITIAL_FEATURES.shape[0]):
            self.INITIAL_FEATURES[k ,:] = state
        for t in range(0, len(test) - 1, self.skip):
            action, last_state = self.sess.run([self.logits ,self.last_state],
                                               feed_dict={self.X :[self.INITIAL_FEATURES],
                                                          self.hidden_layer :init_value})
            action, init_value = np.argmax(action[0]), last_state
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= test[t]:
                inventory.append(test[t])
                initial_money -= test[t]
                states_buy.append(t)
                hold_num+=1
                print('day %d: buy 1 unit at price %f, total balance %f '% (t, test[t], initial_money))
                Journal += 'day %d: buy 1 unit at price %f, cash: %f ,hold:%f' % (
                t, self.trend[t], initial_money, hold_num) + "\n"
            elif action == 2 and len(inventory):
                hold_num -= 1
                bought_price = inventory.pop(0)
                initial_money += test[t]
                states_sell.append(t)
                try:
                    invest = ((test[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, test[t], invest, initial_money)
                )
                Journal += 'day %d, sell 1 unit at price %f, investment %f %%, cash %f,hold:%f' % (
                t, close[t], invest, initial_money, hold_num) + "\n"
            cash.append(initial_money)
            holder.append(hold_num)
            new_state = np.append([self.get_state(t + 1)], self.INITIAL_FEATURES[:3, :], axis = 0)
            self.INITIAL_FEATURES = new_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest,cash,holder,Journal

    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        cash=[]
        holder=[]
        hold_num=0
        cash.append(initial_money)
        state = self.get_state(0)
        Journal='Training Trading Log：\n'
        init_value = np.zeros((1, 2 * self.LAYER_SIZE))
        for k in range(self.INITIAL_FEATURES.shape[0]):
            self.INITIAL_FEATURES[k ,:] = state
        for t in range(0, len(self.trend) - 1, self.skip):
            action, last_state = self.sess.run([self.logits ,self.last_state],
                                               feed_dict={self.X :[self.INITIAL_FEATURES],
                                                          self.hidden_layer :init_value})
            action, init_value = np.argmax(action[0]), last_state
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                hold_num+=1
                print('day %d: buy 1 unit at price %f, total balance %f '% (t, self.trend[t], initial_money))
                Journal+='day %d: buy 1 unit at price %f, cash: %f ,hold:%f'% (t, self.trend[t], initial_money,hold_num)+"\n"
            elif action == 2 and len(inventory):
                hold_num -= 1
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, close[t], invest, initial_money)
                )
                Journal+= 'day %d, sell 1 unit at price %f, investment %f %%, cash %f,hold:%f'% (t, close[t], invest, initial_money,hold_num)+"\n"

            cash.append(initial_money)
            holder.append(hold_num)
            new_state = np.append([self.get_state(t + 1)], self.INITIAL_FEATURES[:3, :], axis = 0)
            self.INITIAL_FEATURES = new_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest,cash,holder,Journal


    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            #模型外参初始化
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            #模型内参初始化
            init_value = np.zeros((1, 2 * self.LAYER_SIZE))
            for k in range(self.INITIAL_FEATURES.shape[0]):
                self.INITIAL_FEATURES[k ,:] = state

            for t in range(0, len(self.trend) - 1, self.skip):
                #每个时刻下做出行动
                if np.random.rand() < self.EPSILON:
                    action = np.random.randint(self.OUTPUT_SIZE)
                    #随机探索（在卖，买，停）随机选择
                else:
                    action, last_state = self.sess.run([self.logits,
                                                        self.last_state],
                                                       feed_dict={self.X :[self.INITIAL_FEATURES],
                                                                  self.hidden_layer :init_value})
                    action, init_value = np.argmax(action[0]), last_state

                #next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]

                invest = ((starting_money - initial_money) / initial_money)
                b=self.magicn-1
                new_state = np.append([self.get_state(t + 1)], self.INITIAL_FEATURES[:b, :], axis = 0)
                #扩充记忆池
                self._memorize(self.INITIAL_FEATURES, action, invest, new_state,
                               starting_money < initial_money, init_value[0])
                self.INITIAL_FEATURES = new_state
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE) #训练中最好选择batchsize的量作为训练集但也考虑记忆吃初始化时数据量不足
                replay = random.sample(self.MEMORIES, batch_size)#提取记忆放在reply池
                X, Y, INIT_VAL = self._construct_memories(replay)

                cost, _ = self.sess.run([self.cost, self.optimizer],
                                        feed_dict={self.X: X, self.Y :Y,
                                                   self.hidden_layer: INIT_VAL})
                #这段代码更新了epsilon 随着时间的增长，随机步的可能性将下降
                self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)

            if ( i +1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, cost: %f, total money: %f ' %(i + 1, total_profit, cost,
                                                                                   starting_money))
            saver = tf.train.Saver(max_to_keep=1)
            saver.save(self.sess, "model/RDQNapp/model_rnn.ckpt")

if __name__ == '__main__':
    #数据导入
    #训练数据
    df_full = pd.read_csv('Your root\\META_tr.csv')
    print(df_full.head())
    #检验数据
    df_t_full = pd.read_csv('Your root\\META_test.csv')
    #随机数设定
    tf.disable_v2_behavior()
    random_seed = 100
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    #保存数据和命名
    df= df_full.copy()
    df_t=df_t_full.copy()
    path='Recurrent Q-learning agent'+"_meta_"+str(time.localtime().tm_mday)+"_"+str(time.localtime().tm_hour)+"_"+str(time.localtime().tm_min)
    os.makedirs(path)
    name = path+"\\"+'Recurrent Q-learning agent'+"_meta_"+str(time.localtime().tm_mday)+"_"+str(time.localtime().tm_hour)+"_"+str(time.localtime().tm_min)
    #数据
    close = df.Close.values.tolist()
    test=df_t.Close.values.tolist()
    #初始资金
    initial_money = 1000
    #历史记录时间
    window_size = 20
    drop=20 #这里采用4是为了减少过拟合状态
    #交易步长
    skip = 1


    batch_size = 32
    agent = Agent(state_size = window_size,
                  window_size = window_size,
                  trend = close,
                  skip = skip,
                  mag=drop)


    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)
    states_buy, states_sell, total_gains, invest ,cash , holder,log_tr= agent.buy(initial_money = initial_money)
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.savefig(name+'_train.png')
    a=total_gains
    file = open(name+'_trading_log_train.txt','w')
    file.write(log_tr)
    file.close()

    states_buy, states_sell, total_gains, invest ,cash , holder,log_test= agent.test(initial_money = initial_money,test=test)
    fig = plt.figure(figsize = (15,5))
    plt.plot(test, color='r', lw=2.)
    plt.plot(test, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(test, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.savefig(name+'_test.png')
    b=total_gains
    file = open(name+'_trading_log_test.txt','w')
    file.write(log_test)
    file.close()
    test_re="_train_%f_test_%f"%(a, b)

    os.rename(path, path+test_re)
