#! /usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean as euc

class Person:

    def __init__(self, master, id, ideas, w):
        """Initialize argmunets.

        Keyword arguments:
        master    : Master class (call from "Meeting")
        self.id   : Id for each person [0, 1, ..., N-1]
        self.ideas: ideas in space [0,1] × [0,1]
        self.w    : probability weight for the person to speak
        """
        self.id = id
        self.ideas = ideas
        self.w = w
        # add_ideas
        master.ideas += [[(i1, i2), [self.id, 0]] for i1, i2 in self.ideas]


class Meeting:

    def __init__(self, K, N, S=20, r=0.07, draw=True):
        self.K = K
        self.N = N
        self.S = S
        self.r = r
        self.ideas = []
        self.minutes = []
        self.draw = draw

    def gather_people(self, ideass=None, weights=None):
        """Gather participants.

        Keyword arguments:
        ideas  : list of ideas for each person
               ex) [((0.3,0.1),(0.2,0.5)), ((0.5,0.6))] when N = 2
        weights: list of weights for the probability of the person to speak
        """
        if not ideass:
            x = np.random.rand(self.N, self.S*2)
            ideass = []
            for _x in x:
                ideass.append([(i,j) for i,j in zip(_x[::2], _x[1::2])])
        if not weights:
            weights = [1.] * self.N
        for i, ideas, w in zip(range(self.N), ideass, weights):
            Person(self, i, ideas, w)

    def clustering(self):
        self.cluster_link = []
        cell_num = int(1./self.r)
        lr = 1./cell_num

        cell = [[[]]*cell_num]*cell_num
        rcell = []
        for i, idea in enumerate(self.ideas):
            cellx = int(idea[0][0]/lr)
            celly = int(idea[0][1]/lr)
            cell[cellx][celly].append(i)
            rcell.append((cellx, celly))

        def uniq_list(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if x not in seen and not seen_add(x)]

        def find_nearest(cx, cy, idea_id, num):
            """

            cx, cy: the cell number which contains the idea
            idea_id: index in self.ideas
            """
            place = self.ideas[idea_id][0]
            tmp = []
            nearest = []
            cid = []
            for i in uniq_list([max(0, cx - 1), cx, min(cx + 1, cell_num - 1)]):
                for j in uniq_list([max(0, cy - 1), cy, min(cy + 1, cell_num - 1)]):
                    tmp += cell[i][j]
            for i in tmp:
                if euc(self.ideas[i][0], place) < self.r:
                    nearest.append(i)
                    prenum = self.ideas[i][1][1]
                    if prenum == 0:
                        cid.append(num)
                        self.cluster_link.append((idea_id, i))
                    elif prenum <= num:
                        cid.append(prenum)
                        self.cluster_link.append((idea_id, i))
            if len(nearest) == 0:
                self.ideas[idea_id][1][1] = cluster_id
                return 1
            else:
                cluster_id = min(cid)
                if cluster_id < num:
                    ans = 0
                else:
                    ans = 1
                for i in nearest:
                    self.ideas[i][1][1] = cluster_id
                return ans

        num = 1
        for i in range(len(self.ideas)):
            cx, cy = rcell[i]
            num += find_nearest(cx, cy, i, num)

    def init(self):
        self.gather_people()
        self.clustering()
#        self.subject = np.random.rand(self.a)
#        self.minutes.append((self.subject, self.N))
        if self.draw:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            self.fig = plt.figure(figsize=(9, 9))
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.labels = ['subject']
            self.s1 = []
            for idea, tag in self.ideas:
                x = idea[0]
                y = idea[1]
                s = self.ax.scatter(
                    x, y, c=colors[tag[0]%len(colors)], alpha=0.2)
                self.labels.append(str(i))
                self.s1.append(s)
            data = []
            for link in self.cluster_link:
                ix = self.ideas[link[0]][0][0]
                iy = self.ideas[link[0]][0][1]
                jx = self.ideas[link[1]][0][0]
                jy = self.ideas[link[1]][0][1]
                data += [(ix, jx), (iy, jy), 'k']
            self.ax.plot(*data)

    def progress(self):
        self.init()
        preidea = self.subject
        self.k = 1

        while self.k < self.K + 1:
            # l: (distance, speaker, idea_id) list for who can speak
            l = []
            for person_id, person in enumerate(self.members):
                # chosed: (distance, idea_id)
                chosed = person.chose_idea(preidea)
                if chosed:
                    l.append((chosed[0], person_id, chosed[1]))
            # if no one can speak: meeting ends.
            if len(l) == 0:
                print "no one can speak."
                break
            i = [(person_id, idea_id)
                 for distance, person_id, idea_id in sorted(l)]

            for person_id, idea_id in i:
                rn = np.random.rand()
                if rn < self.members[person_id].p:
                    idea = self.members[person_id].ideas.pop(idea_id)
                    self.minutes.append((idea, person_id))
                    if self.case == 2:
                        preidea = idea
                    self.callback()
                    self.k += 1
                    break
            else:
                self.minutes.append((self.subject, self.N))
                self.callback()
                self.k += 1

        self.after()

    def callback(self):
        if self.draw:
            if self.minutes[-1][1] == self.N or self.minutes[-2][1] == self.N:
                alpha = 0.2
            else:
                alpha = 1.0
            ix = self.minutes[-2][0][0]
            iy = self.minutes[-2][0][1]
            jx = self.minutes[-1][0][0]
            jy = self.minutes[-1][0][1]
            l1 = self.ax.plot([ix, jx], [iy, jy], color='black', alpha=alpha)
            self.ax.text(jx, jy, '%d' % self.k, fontsize=20)
        else:
            pass

    def after(self):
        if self.draw:
            plugins.connect(
                self.fig, plugins.InteractiveLegendPlugin(
                    self.s1, self.labels, ax=self.ax))
            mpld3.show()
        else:
            print meeting.minutes

if __name__ == '__main__':

    trial = 20
    r = np.linspace(0.01, 0.3, num=100)
    phi = []
    for _r in r:
        _phi = 0.
        for t in range(trial):
            meeting = Meeting(K=50, N=6, r=_r, draw=False)
            meeting.init()
            _phi += max([x[1][1] for x in meeting.ideas])/float(len(meeting.ideas))
        phi.append(_phi/trial)
    plt.plot(r, phi)
    plt.show()
