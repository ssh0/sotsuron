#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, December 2014.

import matplotlib.pyplot as plt
import numpy as np
import time
import operator

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


class Person(object):

    def __init__(self, ideas_num=10, place=(0., 0.), **kwargs):
        # 意見は0~1の間の値を一様に取りうる
        self.ideas = list(np.random.random(ideas_num))
        # 発言者の実際の位置が2次元の座標として表せる
        self.place = place
        # その他の特徴量
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def distance(self, p):
        # 人pと自分との間の距離(ユークリッド距離)
        d = np.sqrt((self.place[0]-p.place[0])**2 + (self.place[1]-p.place[1])**2)
        return d


class meeting(object):

    def __init__(self, N):
        # 会議の参加人数
        self.N = N
        # 意見の時系列
        self.ideas = []
        # 発言者の時系列
        self.speaker = []
        # 時刻
        self.k = 0
        # 張られたリンク(時刻, 時刻)のタプルで表現する
        self.links = []
        # リンクの数(各時刻)
        self.l = [0]
        # リンクの数(累計)
        self.L = [0]
        # はじめの意見(議題)
        x0 = 0.
        self.ideas.append(x0)
        # 議題は沈黙の意見だとする
        self.speaker.append(0)
        # 会議のメンバーとして沈黙を加える
        silent = Person(place=(0., 0.))
        silent.ideas = [0.]
        self.members = {0: silent}
        # 沈黙を中心とした円周上に等間隔で参加者が存在する
        r = 3.
        deg = np.linspace(0., 360., self.N, endpoint=False)
        for n in range(1, self.N+1):
            rad = np.radians(deg[n-1])
            self.members[n] = Person(place=(r*np.cos(rad), r*np.sin(rad)))
        self.radius = 1/3. # 意見の近さは|x-y|の期待値にする

    def p(self, i):

        # 発言者の物理的距離に対する関数
        def g(x):
            return 1./(1.+x)

        # 参加者の中で話せる人のみを対象に
        _N = [0]
        for k in range(1, self.N+1):
            if len(self.members[k].ideas):
                _N.append(k)

        # それらの人たちに対し、関数gによる重み付けの確率を付与
        w = []
        for n in _N:
            d = self.members[n].distance(i)
            w.append(g(d))
        w = np.array(w)
        sum_ = np.sum(w)
        _p = list(w/sum_)
        p = list(accumulate(_p))
        rn = np.random.rand()
        nm = 0
        while True:
            if p[nm] > rn:
                break
            else:
                nm += 1
        # その確率で選ばれた人の名前を返す
        j = _N[nm]
        return j

    def q(self, j):
        # 発言者jが選ばれた時、持っている意見から等確率で意見を取り出す
        x_j = self.members[j]
        if j == 0:
            return x_j.ideas[0]
        else:
            return x_j.ideas.pop()

    def distance(self, x, y):
        # 意見の近さを絶対値で表現
        d = np.abs(x - y)
        if d == 0:
            return self.radius + 1
        else:
            return d

    def connect(self):
        l = 0
        for i, v in enumerate(self.ideas[:-1]):
            # k番目の意見と意見が近い時、それらノードの間にリンクを形成する
            if self.distance(v, self.ideas[self.k]) < self.radius:
                self.links.append((i, self.k))
                l += 1
        return l

    def check_agreement(self):
        # 合意チェック 参加人数Nによる関数
        def L(N):
            return N**2
        print 'link: ', self.l[-1]
        if self.l[-1] > L(self.N):
            return True
        else:
            return False

    def check_ideas(self):
        for k in range(1, self.N+1):
            if len(self.members[k].ideas):
                return True
        return False

    def f_L(self):
        # リンクから会議の評価
        # 単純に会議終了時に得られたリンクの数を返す
        return self.L[-1]


    def f_T(self):
        # 会議に必要な時間の評価
        # 単純に必要な時間kを返す
        return self.k

    def f(self):
        # f_Lとf_Tを使った評価関数f
        return self.f_L() - self.f_T()

    def end(self):
        # 会議の通常終了、各定義量の計算や受け渡しなどはここで
        plt.ioff()
        print 'self.k: ', self.k
        print 'self.L: ', self.L
        print 'self.links: ', self.links
        print 'self.ideas: ', self.ideas
        print 'self.speaker: ', self.speaker
        plt.show()

    def end2(self):
        # 会議の異常終了(発言者が発言できなくなる)
        pass

    def progress(self):
        x = [i.place[0] for i in self.members.values()]
        y = [i.place[1] for i in self.members.values()]
        plt.scatter(x, y)
        plt.ion()
        plt.draw()
        while True:
            j = self.p(self.members[self.speaker[-1]])
            print 'speaker:', j
            self.ideas.append(self.q(j))
            self.speaker.append(j)
            self.k += 1
            self.l.append(self.connect())
            self.L.append(len(self.links))
            plt.plot([self.members[self.speaker[-2]].place[0], self.members[self.speaker[-1]].place[0]], [self.members[self.speaker[-2]].place[1], self.members[self.speaker[-1]].place[1]])
            plt.draw()
            time.sleep(0.5)
            if self.check_agreement():
                self.end()
                break
            if not self.check_ideas():
                self.end2()
                break

if __name__ == '__main__':
    N = 6
    app = meeting(N)
    app.progress()
