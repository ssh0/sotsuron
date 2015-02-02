#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean as euc
import matplotlib.pyplot as plt
import multiprocessing as mp


class Person:

    def __init__(self, S, a, p=0.5):
        self.S = S
        self.a = a
        self.p = p

    def gather(self):
        """make person to participate the meeting.
        """
        self.ideas = self.has_idea()

    def has_idea(self):
        """a person has self.S ideas with self.a dimension.
        """
        return list(np.random.rand(self.S, self.a))

    def chose_idea(self, idea, idea2=None):

        def nearness1(x, y, z):
            """calculate nearness of x for (y, z)
            by calculating a linear combination.
            """
            alpha = 1.
            beta = 1.
            return alpha*euc(x, y) + beta*euc(x, z)

        def nearness2(x, y, z):
            """calculate nearness of x for (y, z)
            by distance between x and the dividing point of (y, z) with t.
            """
            # t > 0
            # t <= 1: interior division
            # t > 1: exterior division
            t = 0.5 
            x, y, z = np.array(x), np.array(y), np.array(z)
            return euc(t*(y-x) + (1.-t)*(z-x), (0., 0.))

        if len(self.ideas) == 0:
            return False
        # return min(d) and its idea_id
        if idea2 == None:
            return min([(euc(vec, idea), idea_id) for idea_id, vec
                        in enumerate(self.ideas)])
        else:
            return min([(nearness1(vec, idea, idea2), idea_id)
                        for idea_id, vec in enumerate(self.ideas)])

class Meeting:

    """Simulate a meeting with "simple3" situation.

    Give keyword arguments:

        K = 20 # Time limit
        N = 6 # a number of participants
        S = 10 # a number of ideas for each participants
        a = 2 # the dimension of an idea
        p = 0.5 # probability that a person speak

    Output:

        self.minutes: list of
                      ( idea(which is vector with a dimension)
                      , who(person_id in the list "self.membes"))
        self.k: stopped time (=len(self.minutes))
    """

    def __init__(self, K=20, N=6, S=10, a=2, p=0.5, case=2):
        self.K = K
        self.N = N
        self.S = S
        self.a = a
        self.p = p
        self.case = case  # case in the above cell: 2, 3, 4 or 5
        if not self.case in [2, 3, 4, 5]:
            raise ValueError
        self.members = []
        self.minutes = []  # list of (idea, who)
        self.k = 0

    def gather_people(self):
        """gather people for the meeting.

        You can edit what ideas they have in here.
        """
        for n in range(self.N):
            person = Person(self.S, self.a, self.p)
            # person.has_idea = some_function()
            # some_function: return list of self.S arrays with dim self.a.
            person.gather()
            self.members.append(person)
        self.members = np.array(self.members) 

    def progress(self):
        """meeting progress
        """
        self.init()
        preidea = self.subject
        prepreidea = None
        self.k = 1

        while self.k < self.K + 1:
            # l: (distance, speaker, idea_id) list for who can speak
            l = []
            for person_id, person in enumerate(self.members):
                # chosed: (distance, idea_id)
                chosed = person.chose_idea(preidea, prepreidea)
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
                    if self.case == 3:
                        preidea = idea
                    elif self.case == 4:
                        prepreidea = idea
                    elif self.case == 5:
                        prepreidea = preidea
                        preidea = idea
                    self.k += 1
                    break
            else:
                self.minutes.append((self.subject, self.N))
                self.k += 1
        self.minutes = np.array(self.minutes)

    def init(self):
        self.gather_people()
        self.subject = np.random.rand(self.a)
        self.minutes.append((self.subject, self.N))

# http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle
# -------------------------------------------------------------------

def myplot(x, y, xfit=np.array([]), yfit=np.array([]), param=None,
            scale=['linear', 'linear', 'log', 'log'], case=[2, 3, 4, 5]):
    """my plot function

    x: {'label_x', xdata}
    xdata: numpy array of array
    y: {'label_y', ydata}
    ydata: numpy array of array
    ydelta: np.array([case 1 delta], [...])
    param: {'a': 10, 'b': 20}
    """
    if param:
        s = [r'$%s = %f$' % (k, v) for k, v in param.items()]
        label = s[0]
        for _s in s[1:]:
            label += ", " + _s
    label_x, xdata = x.items()[0]
    label_y, ydata = y.items()[0]
    if len(scale)%2 == 1:
        raise ValueError("'scale' must be even number")
    fignum = len(scale)/2
    figsize_y = 7 * fignum
    fig = plt.figure(figsize=(10, figsize_y))
    ax = []
    for num in range(fignum):
        ax.append(fig.add_subplot(fignum, 1, num+1))
        for i, data in enumerate(zip(xdata, ydata)):
            baseline = ax[num].plot(data[0], data[1],
                                    label="case: %d" % case[i])
        if len(xfit):
            ax[num].plot(xfit, yfit, label=label)
        ax[num].legend(loc='best')
        ax[num].set_xlabel(label_x)
        ax[num].set_ylabel(label_y)
        ax[num].set_xscale(scale[2*(num)])
        ax[num].set_yscale(scale[2*(num)+1])
    plt.show()

def calc_ave_ang_btw_nodes(case, trial, p):
    ang_btw_nodes = []
    N = 6
    for t in range(trial):
        meeting = Meeting(K=30, N=N, S=50, a=2, p=p, case=case)
        meeting.progress()

        p0 = meeting.minutes[0]
        i = 1
        K = len(meeting.minutes)
        while i < K:
            if meeting.minutes[i][1] != N:
                p1 = meeting.minutes[i]
                break
            i += 1
        else:
            continue
        # とりあえず1つ目の戻りを入れたままでやる
        for i, p2 in enumerate(meeting.minutes[i+1:]):
            if p2[1] != N:
                v1 = p1[0] - p0[0]
                v2 = p2[0] - p1[0]
                ang_btw_nodes.append(180.*angle_between(v1, v2)/np.pi)
                p0 = p1
                p1 = p2
    ang_btw_nodes = np.array(ang_btw_nodes)
    return np.average(ang_btw_nodes)

def wrapper(arg):
    return arg[0](arg[1], arg[2], arg[3])

def p_theta2(p, case=2, trial=100):
    data = np.array()
    ave_ang_btw_nodes = data[:, 0]
    var_ang_btw_nodes = data[:, 1]
    return ave_ang_btw_nodes, var_ang_btw_nodes

if __name__ == '__main__':
    pool = mp.Pool
    proc = mp.cpu_count() * 2

    ydata = []
    case = [2, 3, 4, 5]
    trial = 100
    p = np.linspace(0.01, 1.0, 100)
    for c in case:
        jobs = [(calc_ave_ang_btw_nodes, c, trial, _p) for _p in p]
        ydata.append(np.array(pool(proc).map(wrapper, jobs)))
    myplot({r'$p$': np.array([p]*4)}, {r'$\theta$(degrees)': ydata},
           scale=['linear', 'linear'], case=case)

