# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time
import datetime as dt

class RockScissorsPaper:
    def __init__(self, number_of_data=1000):
        self.number_of_data = number_of_data

    def opponent_hand(self):
        rand_rock = np.random.rand()
        rand_scissors = np.random.rand()
        rand_paper = np.random.rand()
        total = rand_rock + rand_scissors + rand_paper
        return [rand_rock/total, rand_scissors/total, rand_paper/total]

    def winning_hand(self, rock, scissors, paper) -> [float, float, float]:
        mx = max([rock, scissors, paper])
        if rock == mx: return [0, 0, 1]
        if scissors == mx: return [1, 0, 0]
        if paper == mx: return [0, 1, 0]

    def get_supervised_data(self, n_data=None):

        if n_data is None:
            n_data = self.number_of_data

        supervised_data_input = []
        supervised_data_output = []
        for i in range(n_data):
            rock_prob, scissors_prob, paper_prob = self.opponent_hand()
            input_prob = [rock_prob, scissors_prob, paper_prob]
            supervised_data_input.append(input_prob)
            supervised_data_output.append(self.winning_hand(*input_prob))
        return {'input': supervised_data_input, 'output': supervised_data_output}

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('summary-dir', '/temp/tensorflow/summary')
tf.app.flags.DEFINE_integer('max-epoch', 100)
tf.app.flags.DEFINE_integer('batch-size', 10)
tf.app.flags.DEFINE_float('learning-rate', 0.07)
tf.app.falgs.DEFINE_integer('test-data', 1000)
tf.app.falgs.DEFINE_integer('training-data', 1000)
tf.app.flags.DEFINE_boolean('skip-training', False)

def train_and_test(training_data, test_data):
    if len(training_data['input'])!= len(training_data['output']):
        print("トレーニングデータの入力と出力のデータの数が一致しません")
        return
    if len(test_data['input'])!= len(test_data['output']):
        print("テストデータの入力と出力のデータの数が一致しません")
        return

    with tf.name_scope('Inputs'):
        input = tf.placeholder(tf.float32, shape=[None, 3], name='Inputs')
    with tf.name_scope('Outputs'):
        true_output = tf.placeholder(tf.float32, shape=[None, 3], name='Outputs')

    def hidden_layer(x, layer_size, is_output=False):
        name = 'Hidden-Layer' if not is_output else 'Output-Layer'
        with tf.name_scope(name):
            w = tf.Variable(tf.random_normal([x._shape[1].value, layer_size]), name='Weight')

            b = tf.Variable(tf.zeros([layer_size]), name='Bias')

            z = tf.matmul(x, w) + b

            a = tf.tanh(z) if not is_output else z
        return a

    layer1 = hidden_layer(input, 10)
    layer2 = hidden_layer(layer1, 10)
    output = hidden_layer(layer2, 3, is_output=True)

    with tf.name_scope("Loss"):
        with tf.name_scope("Cross-Entropy"):
            error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_output, logits=output))
        with tf.name_scope("Accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(true_output, 1), tf.argmax(output, 1)), tf.float32))*100.0
        with tf.name_scope("Prediction"):
            prediction = tf.nn.softmax(output)

    with tf.name_scope("Train"):
        train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(error)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(FLAGS.summary_dir + '/' + dt.datetime.now().strftime('%Y%m%d-%H%M%S'), sess.graph)
    tf.summary.scalar('CrossEntropy', error)
    tf.summary.scalar('Accuracy', accuracy)
    summary = tf.summary.merge_all()

    def train():
        print('----- 学習開始 ------')
        batch_size = FLAGS.batch_size
        loop_per_epoch = int(len(training_data['input'])/ batch_size)
        max_epoch = FLAGS.max_epoch
        print_interval = max_epoch / 10 if max_epoch >= 10 else 1
        step = 0
        start_time = time.time()
        for e in range(max_epoch):
            for i in range(loop_per_epoch):
                batch_input = training_data['input'][i*batch_size:(i+1)*batch_size]
                batch_output = training_data['output'][i*batch_size:(i+1)*batch_size]
                _, loss, acc, report = sess.run([train_op, error, accuracy, summary], feed_dict={input:batch_input, true_output:batch_output})
                step += batch_size

            writer.add_summary(report, step)
            writer.flush()

            if (e+1) % print_interval == 0:
                learning_speed = (e +1.0) / (time.time() - start_time)
                print('エポック{:3} クロスエントロピー:{:.6f} 正答率:{:6.2f}% 学習速度:{:5.2f}エポック/秒'.format(e+1, loss, acc, learning_speed))

        print('----学習終了----')
        print('{}エポックの学習に要した時間: {:.2f}秒'.format(max_epoch, time.time() - start_time))

    def test():
        print('---検証開始---')
        print('{:5} {:20}  {:20} {:20}  {:2}'.format(",'相手の手', '勝てる手','AIの判断', '結果'"))
        print('{} {:3} {:3} {:3} {:3} {:3} {:3} {:3} {:3} {:3} {:3}'.format('No. ', 'グー', 'チョキ', 'パー' 'グー', 'チョキ', 'パー', 'グー', 'チョキ', 'パー　'))

        def highlight(rock, scissors, paper):
            mx = max(rock, scissors, paper)
            rock_prob_em = '[{:6.4f}]'.format(rock) if rock == mx else '{:^8.4f}'.format(rock)
            scissors_prob_em = '[{:6.4f}]'.format(scissors) if rock == mx else '{:^8.4f}'.format(scissors)
            paper_prob_em = '[{:6.4f}]'.format(paper) if rock == mx else '{:^8.4f}'.format(paper)
            return [rock_prob_em, scissors_prob_em, paper_prob_em]


        win_count = 0
        for k in range(len(test_data['input'])):
            input_probs = [test_data['input'][k]]
            output_probs = [test_data['output'][k]]

            acc, predict = sess.run([accuracy, prediction], feed_dict={input: input_probs, true_output:output_probs})

            best_bet_label = np.argmax(output_probs, 1)
            best_bet_logit = np.argmax(predict, 1)
            result = '外れ'
            if best_bet_label == best_bet_logit:
                win_count += 1
                result = '一致'

            print('{:<5}{:8} {:8} {:8}'.format(*tuple([k+1]+highlight(*input_probs[0]))), end='')
            print(' ', end='')
            print('{:8} {:8} {:8}'.format(*tuple(highlight(*output_probs[0]))), end='')
            print(' ', end='')
            print('{:8} {:8} {:8}'.format(*tuple(highlight(*predict[0]))), end='')
            print('{:2}'.format(result))

        print('--検証終了---')
        print('AIの勝率: {}勝/{}敗 勝率{:4.3f}%'.format(win_count, FLAGS.test_data-win_count, (win_count/len(test_data['input'])* 100.0)))
    print('学習無しの素の状態でAIがじゃんけんに勝てるか確認')
    test()

    if not FLAGS.skip_training:
        train()
        print('学習後、AIのじゃんけんの勝率はいかに・・・！')
        test()

def main(argv=None):
    janken = RockScissorsPaper()
    training_data = janken.get_supervised_data(FLAGS.training_data)
    test_data = janken.get_supervised_data(FLAGS.test_data)

    train_and_test(training_data, test_data)

if __name__ == '__main__':
    tf.app.run()






