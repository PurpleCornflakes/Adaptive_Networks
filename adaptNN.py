import numpy as np
import time

np.random.seed(2017)
class AdaptiveNet:

    def __init__(self, N, M, input_node, desired_node):

        self.N = N
        self.M = M
        self.T = 0.33 # threshold
        self.input_node = input_node
        self.desired_node = desired_node
        self.net = np.zeros((N,M), dtype = np.int)
        self.net[0,input_node] = 1

        # initialize weights
        # every successive 3 in this array corresponds to the 3 lines
        # o o o
        # \ | /
        #   o     
        self.weights = np.random.random((N-1, M*3))
        # normalize
        self.normalize_weight()

    def get_new_net(self):
        new_net = self.net[:]
        tmp = self.__expand(self.net)
        tmp *= self.weights
        bf_threshold = np.asarray([np.sum(tmp[:, j*3:j*3+3], axis = 1) for j in range(self.M)]).T
        # update according to threshold
        new_net[1:,:] = np.where(bf_threshold <= self.T, 0, 1)
        return new_net

    def one_iterate(self):
        new_net = self.get_new_net()
        A = np.sum(new_net[-1,:])
        if A==0 or new_net[-1, self.desired_node] == 1:
            r = 0.01
        else:
            r = -0.1
        self.T += np.sign(A-1)*0.01/self.N

        tmp_old = self.__expand(self.net)

        tmp_new = np.zeros((self.N-1, self.M*3))
        for j in range(self.M):
            tmp_new[:, j*3:j*3+3] = np.array([new_net[1:, [ j,j,j ]]])

        self.weights += r*self.weights*(1-self.weights)*tmp_old*tmp_new
        self.normalize_weight()
        self.net = new_net

    def iterate(self, times, pause_time):
        self.print_out(0)
        for i in range(times):
            time.sleep(pause_time)
            self.one_iterate()
            self.print_out(i+1)
        print("\033[{}B".format(self.N))

    def print_out(self, t):
        conv = lambda x: "\033[01;36m-\033[00m " if x==0 else "\033[01;31mo \033[00m"
        lines = ""
        for i in range(self.N):
            lines += str.join("", map(conv, self.net[i]))+"\n"
        lines += "iter = {0:d}, threshold = {1:.3f}".format(t+1, self.T)+"\n"
        print(lines, end="")
        print("\033[{}A".format(self.N+2))


    def __expand(self, in_array):
        """
        convolveingly expand the in_array, with PBC.
        expand rule: 0,1,2,3,4 --> (4,0,1),(0,1,2),(1,2,3),(2,3,4),(3,4,0)
        """
        tmp = np.zeros((self.N-1, self.M*3), dtype = np.float64)
        for j in range(self.M):
            tmp[:, j*3:j*3+3] = in_array[0:-1, [ (j-1)%self.M, j, (j+1)%self.M ]]
        return np.asarray(tmp, dtype = np.float64)

    def normalize_weight(self):
        for i in range(self.N-1):
            for j in range(self.M):
                jj = 1+j*3
                ww = self.weights[i,[(jj-2)%(self.M*3),jj,(jj+2)%(self.M*3)]]
                self.weights[i,[(jj-2)%(self.M*3),jj,(jj+2)%(self.M*3)]] = ww/np.sum(ww)





if __name__ == "__main__":
    a = AdaptiveNet(64, 64, 32, 8)
    a.iterate(times = 40000, pause_time = 0)


